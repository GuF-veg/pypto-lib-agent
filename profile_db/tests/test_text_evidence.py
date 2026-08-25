# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T2 text-evidence parser parity tests (pure functions, offline)."""

from __future__ import annotations

import json

import pytest

from fixtures import synth_texts
from profile_db.errors import IngestError
from profile_db.ingest.text_evidence import (
    parse_bench_line,
    parse_bytes,
    parse_memory_report,
    parse_perf_hints,
    parse_pmu_csv,
    redact_json,
    redact_paths,
)


def test_parse_perf_hints_matches_fixture() -> None:
    assert parse_perf_hints(synth_texts.PERF_HINTS_TEXT) == synth_texts.PERF_HINTS_EXPECTED


def test_parse_perf_hints_empty_and_blank_lines() -> None:
    assert parse_perf_hints("") == []
    assert parse_perf_hints("  \n\n") == []


def test_parse_bytes_units() -> None:
    assert parse_bytes("32768 B") == 32768
    assert parse_bytes("32KB") == 32768
    assert parse_bytes("192 KB") == 196608
    assert parse_bytes("2MB") == 2 * 1024**2
    assert parse_bytes("junk") is None


def test_parse_memory_report_matches_fixture() -> None:
    assert parse_memory_report(synth_texts.MEMORY_TEXT) == synth_texts.MEMORY_EXPECTED


def test_parse_memory_report_rejects_broken_bytes() -> None:
    with pytest.raises(IngestError, match="cannot parse bytes"):
        parse_memory_report("--- k ---\nVec | uncertain | 512B | 1% | 1\n")


def test_parse_pmu_variant_a() -> None:
    assert parse_pmu_csv(synth_texts.PMU_TEXT_A) == synth_texts.PMU_EXPECTED_A


def test_parse_pmu_variant_b_different_columns() -> None:
    assert parse_pmu_csv(synth_texts.PMU_TEXT_B) == synth_texts.PMU_EXPECTED_B


def test_parse_pmu_rejects_missing_header() -> None:
    with pytest.raises(IngestError, match="exactly one task id"):
        parse_pmu_csv("1000,900\n")


def test_parse_pmu_rejects_non_numeric_counter() -> None:
    bad = "task_id,vec_busy_cycles\n1,oops\n"
    with pytest.raises(IngestError, match="non-numeric"):
        parse_pmu_csv(bad)


def test_parse_pmu_rejects_ambiguous_task_id_columns() -> None:
    bad = "task_id,id,vec_busy_cycles\n1,1,5\n"
    with pytest.raises(IngestError, match="exactly one task id"):
        parse_pmu_csv(bad)


def test_parse_bench_line_from_loop_output() -> None:
    assert parse_bench_line(synth_texts.BENCH_TEXT) == synth_texts.BENCH_EXPECTED


def test_parse_bench_line_minimal_without_rounds() -> None:
    assert parse_bench_line("min=1 median=2 mean=3 max=4") == {
        "min": 1.0,
        "median": 2.0,
        "mean": 3.0,
        "max": 4.0,
        "rounds": None,
    }


def test_parse_bench_line_rounds_as_key() -> None:
    assert parse_bench_line("min=1 median=2 mean=3 max=4 rounds=100") == {
        "min": 1.0,
        "median": 2.0,
        "mean": 3.0,
        "max": 4.0,
        "rounds": 100,
    }


def test_parse_bench_line_rejects_missing_keys() -> None:
    with pytest.raises(IngestError, match="missing"):
        parse_bench_line("min=1 median=2")


def test_redact_paths_recursive() -> None:
    value = {
        "dump_dir": "/home/alice/project/build",
        "nested": ["/data1/alice/x.log", "keep"],
        "root_path": "/root/private",
        "num": 5,
    }
    redacted = redact_paths(value)
    assert redacted["dump_dir"] == "/<redacted>/project/build"
    assert redacted["nested"] == ["/<redacted>/x.log", "keep"]
    assert redacted["root_path"] == "/<redacted>"
    assert redacted["num"] == 5


def test_redact_gufeng_style_home_path() -> None:
    # The cloud layout this repository actually runs on: /data1/home/<user>/...
    assert redact_paths("/data1/home/gufeng/project/pypto-lib-agent") == (
        "/<redacted>/project/pypto-lib-agent"
    )


def test_redact_json_produces_valid_json_without_username() -> None:
    out = redact_json({"p": "/home/mallory/secret"})
    parsed = json.loads(out)
    assert parsed["p"] == "/<redacted>/secret"
    assert "mallory" not in out