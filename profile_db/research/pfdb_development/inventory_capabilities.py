# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inventory current views and channels; never regenerate the frozen A0 tool list.

Run in conda pypto with --output <new.csv>. Existing output is refused. This
exports schemas and audited source contracts without opening a database,
calling a model, rendering a file, or collecting any profiling artifacts.
Test references document related coverage, not an exhaustive execution claim.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
from pathlib import Path

from profile_db.mcp_server import _LIFECYCLE_TOOLS, _TOOL_RENAMES
from profile_db.query import list_queries

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "profile_db/src/profile_db/"
TESTS = "profile_db/tests/"
REVIEW = "runtime/model/client/workload change; failed recovery or identity/validation counterexample"
UNASSESSED = "Exhaustive selector/channel combinations; independent causal value of each view; future runtimes and unseen workloads"
FIELDS = (
    "capability_id", "interface", "kind", "workflow", "mode", "owner_question",
    "channels", "parameter_contract", "semantic_constraints", "default_budget_bytes",
    "input_artifacts", "implementation_sources", "permission", "default_response",
    "full_evidence_access", "caller", "cost_scope", "current_decision",
    "evaluated_scope", "evaluation_evidence", "unassessed_scope", "related_tests",
    "replacement", "review_trigger",
)

# Query -> workflow, inputs, semantic constraint. Defaults/types come from models.
CONTEXT = {
    "runs_list": ("orient", "registered run/trial tables", "rank filters runs; CLI list is an alias; list mean is not the campaign objective"),
    "overview": ("orient", "run/task/artifact tables", "RESOURCE scope is capture_declared_topology (metadata), not independently verified chip maxima; profile timing and unprofiled bench are different domains; declared blocks exclude unknowns; per-engine physical rows/cores are whole-capture counts, not utilization or logical work items"),
    "inventory": ("artifact/provenance", "artifact/modality_status/analysis_snapshot", "analysis_details opts into full saved provenance; missing modality is not zero"),
    "density": ("locate/window", "task_row/time_band", "bands is the display coordinate; occupancy does not establish a bottleneck"),
    "sparse_regions": ("locate/window", "time_band", "returns stored_band_idx; this is not density's display bucket"),
    "why_sparse": ("inspect/window", "stored scheduler analysis and time_band", "exactly one of band or stored_band; band uses the same bands as density; counter integrals have distinct units"),
    "region": ("inspect/window", "task_row/idle_gap", "t1_us must exceed t0_us; family/core restrict rows; overlap is not causation"),
    "core": ("inspect/core", "task_row/idle_gap", "single selected core only; does not answer what other cores did"),
    "idle_window": ("inspect/window", "task/task_row/cpm_path", "after-task end to until-task start; implicit until needs observed-CP successor; implicit engine is opposite producer"),
    "task": ("inspect/task", "task/dep_edge/cpm_path", "task identity and timing; shared query layer masks unobserved level-1 lifecycle endpoints/wall fields as unavailable; raw stored placeholders are unchanged"),
    "tasks": ("locate/task", "task", "family is exact; name is substring; at least one required; on_cpm is observed or static"),
    "deps": ("inspect/dependencies", "dep_edge plus saved deps/wait analyses", "direct requires task_id; reduced/dataflow support whole-run or task filtering; wait ignores task/direction and requires persisted windows [64,128,256]"),
    "subgraph": ("inspect/dependencies", "task/dep_edge", "bounded BFS; depth 1..6 and max_nodes 1..200; external nodes and capped status remain explicit"),
    "why_late": ("inspect/task timing", "task/dep_edge/task_row", "END/FIN/dispatch/receive/start decomposition preserves signed early dispatch; missing level-1 fields unavailable"),
    "why_long": ("inspect/task timing", "task/task_row", "task envelope differs from row lifecycle and pipe activity; family_median_us remains lower-middle order statistic"),
    "rows": ("inspect/physical rows", "task_row", "all matched physical rows; no row pagination; retry budget replaces response"),
    "scheduler": ("inspect/scheduler", "saved scheduler analysis plus scheduler_phase/orch_phase", "task_id selects task rows and nearby phases irrespective of view; run-wide view selects summary/phases/cores/rows"),
    "early_dispatch": ("inspect/dispatch", "task/dep_edge/task_row", "producer policy, eligibility and timestamp proof differ; two-clock-tick tolerance follows runtime"),
    "pmu": ("inspect/counters", "PMU CSV parsed into pmu_counter", "ratio is SUM(counter)/SUM(total_cycles), not average ratios or peak-throughput proof; separate PMU-only attach unsupported"),
    "critical_path": ("locate/path", "saved dedicated runtime critical-path analysis", "observed/static retain runtime definitions; path contribution, gap and task span are not interchangeable"),
    "perf_hints": ("inspect/compiler", "perf_hints.log/perf_hint", "verbatim hints with source/origin; no kernel filter or pagination; retry budget"),
    "memory": ("inspect/compiler", "memory_after_AllocateMemoryAddr.txt/memory_entry", "compile-time buffer usage/limit; not scope occupancy or captured HBM bandwidth"),
    "incore": ("inspect/in-core", "manifest_export.csv, instr_metrics.json and registered exported traces", "events requires kernel; pipe/window/offset/limit select events; overlapping pipes remain distinct"),
    "args_dump": ("inspect/arguments", "args_dump.json and registered binary payload", "values requires seq; offset/limit are physical elements, not logical contiguous TensorSpec traversal"),
    "scope_stats": ("inspect/resources", "scope_stats.jsonl and saved scope analysis", "ring/site filters; high-water upper bound differs from realized exit occupancy"),
    "bench": ("verify/objective", "run/bench_stratum/bench_sample", "default emits summary plus all raw samples; no samples toggle; bootstrap is a separate compare option"),
    "host": ("inspect/host", "registered host logs/phases and saved host analysis", "PID/invocation/HID identities preserved; cpu accepts pid but refuses inv/hid; summary count uses filters but cold/warm/loss totals retain run scope"),
}

RESPONSES = {
    "runs_list": "RUN per run: rank, latest trial binding, topology, timing/counts",
    "overview": "RUN/METRIC/RESOURCE/EXECUTION: identity, graph size, profile/bench timings, declared capture topology and observed row/core counts",
    "inventory": "ARTIFACT/MODALITY statuses; analysis_details additionally saved ANALYSIS provenance",
    "density": "BAND occupancy over display buckets and selected engine",
    "sparse_regions": "Ranked sparse storage bands with stored_band_idx/window/occupancy",
    "why_sparse": "Selected band occupancy and official ready/idle/overhead integrals with units",
    "region": "REGION, intersecting task activity and occupancy gaps within selected window",
    "core": "CORE, selected physical ROW records and idle gaps",
    "idle_window": "WINDOW bounds, OCCUPANCY union/core-time, row and idle detail for selected engine",
    "task": "TASK identity, timing envelope and observed/static path membership",
    "tasks": "Matching TASK instances with full task IDs and family/name/engine/path selectors",
    "deps": "direct: annotated edge tensor metadata; reduced/dataflow: saved graph edges; wait: offline WAIT summary/windows",
    "subgraph": "SUBGRAPH/NODE/DEP bounded neighborhood, external nodes and cap status",
    "why_late": "Signed readiness/FIN/dispatch/receive/start timing segments and evidence limits",
    "why_long": "METRIC_DEFINITION and LONG task envelope, lower-middle family statistic/rank and row extrema",
    "rows": "ROW start/end/dispatch/receive/FIN by physical core and row index",
    "scheduler": "run summary/phases/cores/rows from official snapshot; task scope returns matching SCHED_ROW plus local SCHED/ORCH phases",
    "early_dispatch": "Producer policy/eligibility and full/partial/none/unavailable timestamp proof",
    "pmu": "Per-counter sum/denominator ratio; samples=true additionally provenance-preserving captured samples",
    "critical_path": "Ordered PATH tasks with compute/stall/contribution/gap and official path kind",
    "perf_hints": "PERF_HINT text with original source_path/origin, ordered by seq",
    "memory": "MEMORY kernel/space usage and hardware limit",
    "incore": "summary: kernel export status/metrics; events: INCORE_PAGE/INCORE_EVENT with overlap preserved",
    "args_dump": "metadata: physical argument manifest and scalar values; values: selected physical tensor slice plus next_offset",
    "scope_stats": "summary: paired resource/peak/capacity facts; records: paired lifecycle records",
    "bench": "BENCH min/median/mean/max/rounds plus every stored BENCH_SAMPLE",
    "host": "summary: cold/warm/loss/completeness; rounds/tree/cpu: paged HOST records and HOST_PAGE",
}


def source(obj) -> str:
    return str(Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT)) + "::" + obj.__qualname__


def contract(model) -> str:
    schema = model.model_json_schema()
    properties = {}
    for name, details in schema["properties"].items():
        properties[name] = {k: v for k, v in details.items() if k not in ("title", "description")}
    return json.dumps(dict(required=schema.get("required", []), properties=properties,
                           additionalProperties=schema.get("additionalProperties")),
                      sort_keys=True, separators=(",", ":"))


def query_modes(spec) -> list[str]:
    props = spec.params.model_json_schema()["properties"]
    for field in ("view", "kind"):
        if field in props:
            match = re.fullmatch(r"\^\(([^)]+)\)\$", props[field].get("pattern", ""))
            if not match:
                raise ValueError(f"audit the new {spec.name}.{field} domain")
            return [f"{field}={v}" for v in match[1].split("|")]
    return {
        "inventory": ["analysis_details=false", "analysis_details=true"],
        "pmu": ["samples=false", "samples=true"],
        "why_sparse": ["band+bands", "stored_band"],
        "tasks": ["family selector", "name substring selector"],
    }.get(spec.name, ["default"])


def base(**values) -> dict:
    row = dict.fromkeys(FIELDS, "")
    row.update(dict(current_decision="retain_existing; no deletion or default hiding authorized by this inventory",
                    evaluated_scope="Source audited; related deterministic tests exist; no exhaustive per-view/channel or causal agent-utility claim",
                    evaluation_evidence="profile_db/research/pfdb_development/interface_decisions.csv;profile_db/research/pfdb_development/implementation_review.md",
                    unassessed_scope=UNASSESSED, replacement="Existing authoritative handler/adapter; no duplicate analysis introduced",
                    review_trigger=REVIEW))
    row.update(values)
    return row


def query_rows() -> list[dict]:
    result = []
    for spec in list_queries():
        workflow, inputs, constraints = CONTEXT[spec.name]
        tool = "pfdb." + _TOOL_RENAMES.get(spec.name, spec.name)
        modes = query_modes(spec)
        if spec.name == "scheduler":
            modes = [m + ";scope=" + s for m in modes for s in ("run", "task")]
        for mode in modes:
            result.append(base(
                capability_id=f"query.{spec.name}.{mode}", interface=spec.name, kind="query", workflow=workflow,
                mode=mode, owner_question=spec.owner_question,
                channels=f"CLI=pfdb query {spec.name}; API=ProfileDB.query('{spec.name}', **selectors); MCP={tool}",
                parameter_contract=contract(spec.params), semantic_constraints=constraints,
                default_budget_bytes=spec.default_budget_bytes, input_artifacts=inputs,
                implementation_sources=source(spec.handler) + ";" + source(spec.params) + ";docs/debug-and-tune/pfdb-runtime-integration.md",
                permission="read-only database; registered-file reads for requested binary/event slices; no acquisition",
                default_response=RESPONSES[spec.name] + "; registered defaults in parameter_contract; nondefault mode is opt-in",
                full_evidence_access="API/MCP budget_bytes or CLI --budget; on truncation retry same selectors and replace prior response; args/events additionally offset/limit",
                caller="operator optimization agent; diagnostic caller",
                cost_scope="Response budget does not cap rows read internally; full matched evidence may be materialized before prefix truncation",
                related_tests=TESTS + "test_query.py;" + TESTS + "test_query_transport.py;" + TESTS + "test_runtime_integration.py",
            ))
            if (spec.name == "host" and mode != "view=tree") or (spec.name, mode) in (
                ("scope_stats", "view=summary"), ("args_dump", "view=values"), ("incore", "view=events"),
            ):
                result[-1]["evaluated_scope"] = "Synthetic E3 positive packet/export checks exist for this view; no claim of real optimization benefit or independent per-view causal utility"
                result[-1]["evaluation_evidence"] += ";profile_db/research/pfdb_development/prepare_rare_tasks.py;profile_db/research/pfdb_development/e3-capability-study.md"
                result[-1]["related_tests"] += ";profile_db/research/pfdb_development/test_rare_tasks.py"
                result[-1]["unassessed_scope"] += "; corresponding real saved-artifact optimization utility"
    return result


def lifecycle_rows() -> list[dict]:
    cli = dict(attach="attach", refresh="refresh", compare="compare", baseline_diff="baseline diff",
               baseline_list="baseline list", baseline_add="baseline add", register_trial="trial register",
               bind_trial="trial bind", attach_bench="trial attach-bench", set_verdict="trial verdict",
               list_trials="trial list", note="note")
    notes = {
        "attach": "Exactly one run_id (auxiliary artifacts; PMU-only unsupported) or trial_id (immutable candidate manifest; no copy); candidate hashes/scope checked; no collection",
        "refresh": "Explicit saved-analysis rebuild; optional relocated core source; failure preserves prior successful analyses and lifecycle memory",
        "compare": "Capture/profile gate only; workload/validation not checked; family strings are exact, '*' all; bootstrap requires >=3 matched raw strata and estimates mean only",
        "baseline_diff": "Named/latest baseline reuses compare; bootstrap retains mean estimator and raw-strata prerequisites",
        "baseline_list": "Baseline references preserve runs from prune; names do not establish numerical correctness",
        "baseline_add": "Default bench_mean_us comes from run; API additionally accepts criteria mapping not exposed by CLI/MCP; not a median objective contract",
        "register_trial": "Caller hypothesis/files/parent; no profiling collection or automatic optimization decision",
        "bind_trial": "Existing run required; same run idempotent; different run needs child trial; candidate manifests additionally require exact core-capture hashes; producer claims are not correctness certification",
        "attach_bench": "Explicit summary only in API/MCP; CLI also accepts logs but stores summary, not trial raw strata; partial explicit summary never borrows missing run fields",
        "set_verdict": "Caller verdict/evidence_refs; no PASS gate or run required; compile_error supported; does not prove correctness",
        "list_trials": "Default history: unchecked recorded candidate identity; trial_id: current source/build/scope/artifact checks; optional source/scope SHA guards; diagnostics view requires trial_id and source SHA, reads only hash-verified registered diagnostic tails (16 KiB/file); mismatch/stale suppress content/benchmark; missing observations unavailable; claims not certified",
        "note": "Free-text caller memory; not verified measurement or workload contract",
    }
    modes = dict(attach=["link (copy=false)", "copy archive (copy=true)", "trial candidate manifest"],
                 refresh=["registered inputs", "relocated source"],
                 compare=["run delta", "one family", "all families", "bootstrap mean interval"],
                 baseline_diff=["delta", "bootstrap mean interval"], list_trials=["all", "active_only=true", "trial_id candidate evidence", "view=diagnostics;trial_id+source_sha256"])
    result = []
    for name, (model, question, writes) in _LIFECYCLE_TOOLS.items():
        for mode in modes.get(name, ["default"]):
            result.append(base(
                capability_id=f"lifecycle.{name}.{mode}", interface=name, kind="lifecycle",
                workflow="artifact maintenance" if name in ("attach", "refresh") else "experiment memory" if writes or name in ("list_trials", "baseline_list") else "verify/comparison",
                mode=mode, owner_question=question,
                channels=f"CLI=pfdb {cli[name]}; API=ProfileDB.{name}; MCP=pfdb.{name}",
                parameter_contract=contract(model), semantic_constraints=notes[name],
                default_budget_bytes="n/a: write result" if writes else "4096; compare with family=16384",
                input_artifacts="Existing auxiliary artifacts" if name == "attach" else "Existing registered inputs" if name == "refresh" else "Registered run/bench/lifecycle tables; caller-supplied metadata",
                implementation_sources=PACKAGE + "api.py::ProfileDB." + name + ";" + source(model) + ";" + PACKAGE + "mcp_server.py::_lifecycle_result;" + PACKAGE + "cli.py",
                permission="database writer; MCP serve --writable; source artifacts read-only" if writes else "read-only database; no acquisition",
                default_response="Operation result/IDs; writes do not expose MCP budget override" if writes else "Evidence-tagged bounded facts; API returns Result before caller serialization",
                full_evidence_access="Explicit operation; no implicit collection" if writes else "MCP budget_bytes / CLI --budget / API format_result budget; retry replaces prior output",
                caller="data maintainer" if name in ("attach", "refresh") else "operator optimization agent or experiment reviewer",
                cost_scope="Explicit transactional mutation" if writes else "Stored-result read; bootstrap CPU work is opt-in",
                related_tests=TESTS + "test_lifecycle.py;" + TESTS + "test_query_transport.py;" + TESTS + "test_runtime_integration.py",
            ))
    return result


# Only combinations implemented by render.renderers and runtime_adapter.exports.
RENDER_PAIRS = [(k, "png") for k in ("whole", "window", "task", "core")] + [
    ("deps", "html"), ("scope", "html"),
] + [(k, "perfetto") for k in ("whole", "window", "task", "core", "host", "incore", "merged")]


def render_rows() -> list[dict]:
    from profile_db.mcp_server import RenderToolParams

    result = []
    for kind, fmt in RENDER_PAIRS:
        constraints = {
            "whole": "run_id; optional Perfetto time/task/core filters",
            "window": "PNG requires t0_us<t1_us; Perfetto permits optional bounds (without bounds it is unfiltered)",
            "task": "task_id required; PNG has task/dependency view; Perfetto filters official task events",
            "core": "core_index required; PNG includes gaps; Perfetto requires an official matching lane",
            "deps": "Registered deps; official Graphviz dependency must be available",
            "scope": "Registered scope_stats; runtime must find paired resources",
            "host": "Successful saved host analysis and unchanged registered host inputs",
            "incore": "kernel required; registered exported incore_trace:<kernel>",
            "merged": "At least two distinct run_ids, numeric distinct ranks, shared host_clock_domain_id, matching dispatch identities and attached host logs; no device replay",
        }[kind]
        result.append(base(
            capability_id=f"render.{kind}.{fmt}", interface="render", kind="render", workflow="developer visualization/export",
            mode=f"kind={kind};format={fmt}", owner_question=f"Inspect/export {kind} evidence as {fmt}",
            channels=f"CLI=pfdb render {kind} --file-format {fmt}; API=ProfileDB.render('{kind}', run_id, format='{fmt}'); MCP=pfdb.render",
            parameter_contract=contract(RenderToolParams), semantic_constraints=constraints + "; CLI uses --run/--t0/--t1/--core, MCP/API run_id/t0_us/t1_us/core_index; API additionally render_dir and cache/image byte caps; unsupported pairs are not capabilities",
            default_budget_bytes="metadata=4096; MCP optional budget_bytes; image/file and cache caps are separate",
            input_artifacts="DB timing/dependency tables" if fmt == "png" else "Registered immutable runtime export inputs; saved snapshots where applicable",
            implementation_sources=PACKAGE + "api.py::ProfileDB.render;" + PACKAGE + "render/renderers.py;" + PACKAGE + "render/exports.py;" + PACKAGE + "runtime_adapter/exports.py",
            permission="read-only database and source artifacts; writes/evicts only export/cache files; no acquisition",
            default_response="IMAGE metadata plus PNG; MCP includes image bytes" if fmt == "png" else "FILE metadata/reference; does not inline full HTML/trace",
            full_evidence_access="Read returned image/file reference; errors remain unavailable; default format is PNG so HTML/Perfetto must be explicit",
            caller="operator optimization agent; human diagnostic reviewer",
            cost_scope="Render/export CPU and cache I/O; Graphviz for deps HTML; no hardware; byte/cache caps independent of text budget",
            related_tests=TESTS + "test_render.py;" + TESTS + "test_runtime_integration.py;" + TESTS + "test_mcp_server.py",
            unassessed_scope=UNASSESSED + "; real host/scope/in-core/merged utility beyond synthetic fixtures; independent visual-modality optimization benefit",
        ))
    return result


def maintenance_rows() -> list[dict]:
    entries = [
        ("init", "create/migrate", "pfdb init [--path]", "ProfileDB(path, read_only=False); ProfileDB.memory()", "path; API read_only=false default", "DB writer; creates/migrates schema; memory mode has no file", "No source capture required; existing schema migration preserves observations/lifecycle", "db.py;schema/__init__.py", "test_cli.py;test_schema_migrations.py"),
        ("ingest", "link", "pfdb ingest source", "ProfileDB.ingest(source, copy=False, **meta)", "source; program/platform/device/captured_at/notes/tags/rank; modality requests; API runtime_cfg/git metadata; prune_after=true/prune_keep=3", "DB writer; links immutable artifacts; auto-prune by default", "Core records/deps/name-map required; no capture execution; --no-prune disables auto-prune; API accepts custom prune_keep", "ingest/__init__.py;ingest/source.py;cli.py", "test_ingest.py;test_ingest_real_capture.py;test_lifecycle.py"),
        ("ingest", "copy archive", "pfdb ingest source --copy", "ProfileDB.ingest(source, copy=True, **meta)", "same ingest metadata; copy=true", "DB writer and managed artifact-store writes", "Archives registered input files; not a compilation or device collection; file-backed DB required", "ingest/__init__.py;ingest/source.py", "test_lifecycle.py;test_runtime_integration.py"),
        ("register run bench", "summary", "pfdb ingest source --bench 'min=... median=... mean=... max=... rounds=...'", "ProfileDB.ingest(source, bench_min_us=..., bench_median_us=..., bench_mean_us=..., bench_max_us=..., bench_rounds=...)", "existing capture plus bench summary; --bench mutually exclusive with --bench-log", "DB writer; registration only", "Summary alone has no raw strata; no standalone attach-run-bench API/MCP; run must be ingested/reingested", "cli.py::_run_ingest;ingest/__init__.py", "test_text_evidence.py;test_dogfood_feedback.py"),
        ("register run bench", "raw strata", "pfdb ingest source --bench-log log1 --bench-log log2 ...", "ProfileDB.ingest(source, bench_strata=[...], bench_*_us=...)", "one existing raw log per invocation; source hashes/config/sample membership preserved", "DB writer; reads existing logs, never executes benchmark", "headline raw required; API caller supplies both bench summaries and structured strata; >=3 matched strata only required by bootstrap, not plain compare", "cli.py::_run_ingest;ingest/text_evidence.py;lifecycle/bootstrap.py", "test_dogfood_feedback.py"),
        ("ingest-incore", "compatibility alias", "pfdb ingest-incore source --run ID", "ProfileDB.ingest_incore(source, run_id=ID)", "source; run_id", "DB writer; generic auxiliary attach in link mode", "Delegates to attach; no separate MCP tool (use pfdb.attach); --copy supported via generic attach only", "api.py::ProfileDB.ingest_incore;ingest/__init__.py::ingest_incore;ingest/analysis.py", "test_ingest_modalities.py;test_runtime_integration.py"),
        ("prune", "working set", "pfdb prune --keep 3", "ProfileDB.prune(keep=3)", "keep default=3; latest runs + baseline references + running-trial references retained", "DB writer; deletes old derived/registered data and managed copies, never linked originals", "Completed trial summary preserved before run deletion; large traces not retained forever; no MCP prune tool", "lifecycle/prune.py;lifecycle/working_set.py;api.py::ProfileDB.prune", "test_lifecycle.py"),
        ("trial attach-bench", "existing logs summarized", "pfdb trial attach-bench ID --bench-log log1 --bench-log log2 ...", "No raw-log API; parse externally then ProfileDB.attach_bench(trial_id, summary)", "trial_id; --bench or repeatable --bench-log, mutually exclusive", "DB writer; trial summary only", "Unlike ingest --bench-log, CLI trial logs are flattened into a summary; no trial raw-stratum retention; MCP accepts summary fields only", "cli.py::_bench_summary_from_args;api.py::ProfileDB.attach_bench;lifecycle/trial.py", "test_cli.py;test_lifecycle.py"),
        ("serve", "read-only", "pfdb serve --mcp [--path]", "mcp_server.run_stdio(path, writable=False)", "path optional; writable=false default", "Read-only DB; 33 advertised read tools; direct calls still enforce write guards", "Eight database mutation tools omitted; all 41 remain available with --writable; render may write cache files", "mcp_server.py::run_stdio;cli.py", "test_mcp_server.py;test_runtime_integration.py"),
        ("serve", "writer enabled", "pfdb serve --mcp --writable [--path]", "mcp_server.run_stdio(path, writable=True)", "path optional; explicit writable=true", "DB writer held for session; allows lifecycle mutations; concurrent writer blocked", "Write flag is an availability/permission distinction, not profiling acquisition", "mcp_server.py::run_stdio;db.py", "test_mcp_server.py;test_db.py"),
        ("version", "package", "pfdb --version", "profile_db.__version__", "no selectors", "No DB mutation or acquisition", "Package version differs from schema/analysis/adapter/MCP contract versions", "cli.py::_parser;_version.py;__init__.py", "test_cli.py"),
        ("version", "MCP contract", "No CLI equivalent (pfdb --version reports package)", "mcp_server.TOOL_SCHEMA_VERSION", "MCP pfdb.version has no selectors", "No DB mutation or acquisition", "MCP schema version only; not a runtime pin or package version report", "mcp_server.py::_version_tool;mcp_server.py::_dispatch", "test_mcp_server.py"),
    ]
    result = []
    for name, mode, cli, api, selectors, permission, constraints, sources, tests in entries:
        result.append(base(
            capability_id=f"maintenance.{name}.{mode}", interface=name, kind="maintenance", workflow="channel/version" if name in ("serve", "version") else "experiment memory" if name == "trial attach-bench" else "artifact/database maintenance",
            mode=mode, owner_question=f"Manage existing artifacts/database: {name} ({mode})",
            channels=f"CLI={cli}; API={api}; MCP=" + ("pfdb.version" if mode == "MCP contract" else "not a separate tool; see noted equivalent"),
            parameter_contract=selectors, semantic_constraints=constraints, default_budget_bytes="n/a: operation result or process setup",
            input_artifacts="Existing capture/log/manifest only when required; otherwise database/configuration",
            implementation_sources=";".join(PACKAGE + p for p in sources.split(";")), permission=permission,
            default_response="Operation summary/IDs/status, package version, or MCP stdio process",
            full_evidence_access="Follow query/inventory/lifecycle/render capabilities; maintenance does not replace evidence queries",
            caller="data maintainer; experiment runner; operator agent for explicit experiment-memory operations",
            cost_scope="Local files/DB/CPU only; explicit prune/store side effects as stated; no model or NPU execution",
            related_tests=";".join(TESTS + p for p in tests.split(";")),
        ))
    return result


def rows() -> list[dict]:
    result = query_rows() + lifecycle_rows() + render_rows() + maintenance_rows()
    if len({row["capability_id"] for row in result}) != len(result):
        raise ValueError("duplicate capability identity")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows())
