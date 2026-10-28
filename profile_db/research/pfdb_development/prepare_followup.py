# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export six bounded follow-up questions from preserved artifacts, without collection.

The unchanged run_offline scorer reads cases.json; only question and packet reach
the model. reviewer.json owns treatment mapping. Relations add derived information,
not merely formatting. Contract and metadata questions include negative controls.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from prepare_decision_tasks import ROOT, normalize

BASE = "build_output/pfdb_development/"
PILOT = BASE + "optimization-pilot-v1/"
B = BASE + "swa_manual_capture/"
S = PILOT + "swa_manual-0-baseline/attempt-0/"
C = PILOT + "c4_agent-0-semantic_contract/attempt-0/"
G = BASE + "swa_agent_capture/"
FILES = {
    "contract": ("models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py", "d2cca4d12fac58b50a61262cd8104f04435eb6cdcaf56ea63e101174afafafd2"),
    "scan": (PILOT + "c4_manual-1-baseline/attempt-1/candidate.py", "6fded29273bcc107963a853cbcdefacb62d9e299866928388c55df891921db2e"),
    "fixture": (PILOT + "c4_manual-1-baseline/attempt-1/capture/result_s0_p0.json", "ea6eb5f3b491bf1991321c75de5042d1411e9d207830e68e7c9a568886de2748"),
    "old_source": ("models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py", "a787836fb363c4b560dcb60a625f94df3a6dd5280a298a915135ef3ac1620946"),
    "new_source": (S + "candidate.py", "c6e217bec89e36f7ca9147c7320b305eff5b1e9b4563bfa92f2d9d7ceab80e51"),
    "old_ir": (B + "passes_dump/37_after_MemoryReuse.py", "108a3ffa51a9b58390f5fba5bdad38d9482dc1569613b4ef6c36ac7a30725545"),
    "new_ir": (S + "capture/passes_dump/37_after_MemoryReuse.py", "4e0e1e151ec48efb924f470c43e1ab4aee8059bff6d38ef859cee542f0ff3a8e"),
    "old_hint": (B + "report/perf_hints.log", "61c489f5ca3ad71cd23974f337f5c535e08a6cf25dd9af88447373a7911e033e"),
    "new_hint": (S + "capture/report/perf_hints.log", "8850f3e89aab581bab9a7d9fd2edfa145298e9fb8a9b66a9f6cdd53a686410d2"),
    "old_manifest": (B + "collection.json", "dc99bef66a8b11c1e0ff8c071948ed77705df72bae166e0bb8fccb3be789e80a"),
    "new_manifest": (S + "capture/collection.json", "4fd6bf32aab4668bcfeeaf1541bd064622c075c1fbe092ae87319145dc92610a"),
    "old_meta": (B + "compiled_meta.json", "a818d7044dfea0a1e694286ef88aa534e389d2b97e483fb339b56228a2534ac0"),
    "new_meta": (S + "capture/compiled_meta.json", "a818d7044dfea0a1e694286ef88aa534e389d2b97e483fb339b56228a2534ac0"),
    "c4_old_source": ("benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_dspark/prefill_compressor_ratio4_v1.py", "4c7151590d7406116601949bc0604aaac4ee519c078a5c905b89f709a08196ba"),
    "c4_source": (C + "candidate.py", "525d0eee57cf2886667b2c2092cb37b59d50e190c34f6252193b646c62b2cab2"),
    "c4_orch": (C + "capture/orchestration/_compressor_ratio4.cpp", "3e8e877e0d4bf31e4c66cd2b4b77f2e0e0fd6c0d6d126a30e29fd99ce9eaa074"),
    "c4_manifest": (C + "capture/result_s0_p0.json", "6cce889e4ba5268c09084ffaf05ce403751af106091e12f79b2fcda367f7dd91"),
    "pool_source": (PILOT + "swa_agent-0-semantic_contract/source.py", "0ed4e4bac3be06bd8b0b08b66ca49594435e6553180ff5cba3da9546200ff89b"),
    "pool_hint": (G + "report/perf_hints.log", "26ee1f338b87ecbaf370169cb7c58dc1699aac8267f839e8261c63d942c4e2b9"),
    "pool_kernel": (G + "kernels/aic/o_proj_a_quant_aic.cpp", "abf6fc025f2a37d9454590262c5b6bbe6d5f6040e89e63d7b92be9c4d82352e2"),
    "spaces": ("docs/pypto-language/03-types.md", "73f335168c89fab846a66fa9d696def9b7f203c50b56cf3c0d4e3c2405211b23"),
}
ARMS = ("D0", "D1", "R")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> dict[str, str]:
    """Only these frozen, real inputs are accepted; never discover a newer build."""
    snapshots = {}
    for name, (relative, expected) in FILES.items():
        path = ROOT / relative
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"input digest mismatch: {relative}")
        snapshots[name] = path.read_text()
    return snapshots


def artifact(name: str, **fields) -> dict:
    relative, expected = FILES[name]
    return dict(rec="ARTIFACT", source=relative, sha256=expected, evidence="measured", **fields)


def bounded(inputs: dict, name: str, first: int, last: int) -> dict:
    lines = inputs[name].splitlines()
    if not 1 <= first <= last <= len(lines):
        raise ValueError(f"invalid bounded excerpt: {name}:{first}-{last}")
    return artifact(name, first_line=first, last_line=last,
                    text=normalize("\n".join(lines[first - 1:last])))


def record(inputs: dict, name: str, keys: tuple[str, ...]) -> dict:
    obj = json.loads(inputs[name])
    return artifact(name, json_fields=normalize({key: obj[key] for key in keys if key in obj}),
                    absent_json_fields=[key for key in keys if key not in obj])


def hints(inputs: dict, name: str, limit: int | None = None) -> dict:
    matches = [dict(line=i, text=normalize(line)) for i, line in enumerate(inputs[name].splitlines(), 1)
               if "PH-MR-001" in line]
    return artifact(name, scan="all lines containing PH-MR-001", matching_records=len(matches),
                    excerpt_records=matches if limit is None else matches[:limit])


def group(facts: list[dict], kind: str = "existing_artifacts") -> dict:
    return dict(query=kind, params={}, facts=facts)


def scoped_selection(name: str, missing: list[str]) -> dict:
    return group([artifact(name, payload="digest of selected source only"), dict(
        rec="PACKET_SCOPE", evidence="derived", scope="historical artifact groups that follow",
        not_supplied_for_selected_source_in_historical_groups=missing,
        limits="A historical artifact is not reassigned to the selected source. Later groups, if supplied, are separate evidence.",
    )], "artifact_scope")


def relation(fields: dict) -> dict:
    return group([dict(rec="RELATION", evidence="derived", **fields)], "source_anchored_relation")


def definitions(inputs: dict) -> list[dict]:
    """A small explicit case list, rather than an IR parser or diagnostic engine."""
    manifest_keys = ("source_sha256", "reference_sha256", "passed", "fixture", "swimlane", "pmu")
    contract_raw = [group([
        bounded(inputs, "contract", 498, 529), bounded(inputs, "contract", 659, 679),
        bounded(inputs, "contract", 774, 815), bounded(inputs, "scan", 275, 288),
        record(inputs, "fixture", ("source_sha256", "passed", "tokens", "start", "swimlane", "pmu")),
        dict(rec="PACKET_SCOPE", evidence="derived", established_general_position_domain_policy="not supplied",
             new_gap_device_validation="not supplied", limits="The preserved PASS is for the initializer-generated fixture, not every tensor value of that shape."),
    ])]
    contract_refs = ["q1.1", "q1.2", "q1.3", "q1.4"]
    cases = [dict(
        base="contract_fixture", raw=contract_raw, extra=[], required=["q1"], required_current=["q1"],
        question="Consider the saved PASS only as a fixed affine-position fixture, not general deployment authority. "
        "For the bounded scan, return affine_positions_suffice (boolean: tile positions p[j]=p[0]+j), "
        "arbitrary_consistent_paging_breaks_affine_equivalence (boolean, original mapping guards retained), "
        "and saved_pass_proves_general_domain (boolean). State the missing general-scope obligation in next_experiment.",
        expected=dict(affine_positions_suffice=True, arbitrary_consistent_paging_breaks_affine_equivalence=False,
                      saved_pass_proves_general_domain=False),
        derived=relation(dict(inputs=contract_refs, predicate="Under p[j]=p[0]+j the position window is exactly the candidate's physical-index interval; the retained mapping guard filters the same members in both scans.",
                              boundary="This conditional set equivalence and one saved fixture PASS do not establish the general position domain."))),
        dict(
        base="contract_general", raw=copy.deepcopy(contract_raw), extra=[], required=["q1"], required_current=["q1"],
        question="Review the same scan for general operator use. Return initializer_establishes_general_domain "
        "(boolean), monotonic_positions_alone_suffice (boolean), gap_business_legality (established or unresolved), "
        "and new_gap_device_validation (available or unavailable). A static skipped-position diagnostic may test a "
        "premise; distinguish it from an owner input-domain policy and from a supplied device result.",
        expected=dict(initializer_establishes_general_domain=False, monotonic_positions_alone_suffice=False,
                      gap_business_legality="unresolved", new_gap_device_validation="unavailable"),
        derived=relation(dict(inputs=contract_refs, predicate="arange is initializer code, not a general caller contract. Increasing positions with a gap need not satisfy position-minus-first-position = physical index, so the interval inverse is not generally justified.",
                              boundary="No owner policy for gaps or new gap-device outcome is supplied."))),
    ]
    old = [scoped_selection("new_source", ["perf_hints", "MemoryReuse", "validation", "physical_timing"]),
           group([bounded(inputs, "old_source", 60, 72), record(inputs, "old_manifest", manifest_keys), hints(inputs, "old_hint")]),
           group([bounded(inputs, "old_ir", 153, 236)])]
    new = [group([bounded(inputs, "new_source", 60, 72), record(inputs, "new_manifest", manifest_keys), hints(inputs, "new_hint")]),
           group([bounded(inputs, "new_ir", 153, 201)])]
    swa_expected = dict(baseline_warning_count=4, selected_warning_count=0, baseline_right_allocations=2,
                        selected_right_allocations=2, baseline_outer_chunk_association="same_pair",
                        selected_outer_chunk_association="distinct_per_chunk", measured_overlap=False,
                        historical_warning_applies_to_selected=False)
    d0_swa = swa_expected | dict(selected_warning_count="unavailable", selected_right_allocations="unavailable",
                               selected_outer_chunk_association="unavailable")
    cases.append(dict(
        base="swa_fresh", raw=old, extra=new, required=["q1", "q2", "q3"], required_current=["q4", "q5"],
        question="Compare the historical K256 build with the selected K128 source. Return baseline_warning_count "
        "and selected_warning_count (PH-MR-001 record counts), baseline_right_allocations and selected_right_allocations "
        "(proj_a_mm allocation counts), baseline_outer_chunk_association and selected_outer_chunk_association "
        "(same_pair or distinct_per_chunk for Right storage used by adjacent OUTER K chunks). Use unavailable for "
        "missing selected evidence. Also return measured_overlap and historical_warning_applies_to_selected "
        "(booleans). Allocation count and absence of warnings alone are not overlap measurements.",
        expected=swa_expected, expected_d0=d0_swa,
        derived=relation(dict(inputs=["q2.2", "q3.0", "q4.1", "q4.2", "q5.0"],
            historical_edges=["outer chunk kb -> mem_mat_5 -> mem_right_9 and mem_right_11", "outer chunk kb+1 -> mem_mat_7 -> the same mem_right_9 and mem_right_11"],
            selected_edges=["outer chunk kb -> mem_mat_5 -> mem_right_10", "outer chunk kb+1 -> mem_mat_8 -> mem_right_11"],
            boundary="Both proj_a_mm builds allocate two Right buffers. This association change and warning count 4 to 0 are static compiler evidence; selected collection has swimlane=0 and pmu=0, so no measured overlap is established."))))
    identity_raw = [group([artifact(key, file_bytes=len((ROOT / FILES[key][0]).read_bytes())) for key in
                           ("old_meta", "new_meta", "old_source", "new_source", "old_ir", "new_ir")]),
                    group([record(inputs, key, ("schema", "supported_execution_modes", "num_return_types", "platform", "backend_type"))
                           for key in ("old_meta", "new_meta")])]
    cases.append(dict(
        base="identity", raw=identity_raw, extra=[group([bounded(inputs, "old_source", 60, 72), bounded(inputs, "new_source", 60, 72),
                                                        bounded(inputs, "old_ir", 153, 170), bounded(inputs, "new_ir", 153, 166)])],
        required=["q1"], required_current=["q1"],
        question="Use the preserved byte fingerprints for the two builds. Return metadata_bytes_identical, "
        "source_bytes_identical, memory_reuse_bytes_identical, and metadata_hash_sufficient_for_build_match "
        "(booleans). Does a signature/platform artifact alone identify the generating source and IR?",
        expected=dict(metadata_bytes_identical=True, source_bytes_identical=False, memory_reuse_bytes_identical=False,
                      metadata_hash_sufficient_for_build_match=False),
        derived=relation(dict(inputs=["q1.0", "q1.1", "q1.2", "q1.3", "q1.4", "q1.5"],
            edges="The old/new metadata SHA values are equal while the source SHA and MemoryReuse SHA values each differ. This is a concrete many-builds-to-one-signature collision, not a fabricated mismatched file.",
            boundary="Matching requires preserved producer/source/build evidence; equal compiled_meta bytes alone do not supply it."))))
    c4_raw = [scoped_selection("c4_source", ["candidate source body", "candidate orchestration", "candidate physical rows"]),
              group([bounded(inputs, "c4_old_source", 45, 83), bounded(inputs, "c4_old_source", 157, 179)])]
    c4_extra = [group([bounded(inputs, "c4_source", 45, 83), bounded(inputs, "c4_source", 157, 181)]),
                group([bounded(inputs, "c4_orch", 118, 152),
                       record(inputs, "c4_manifest", ("source_sha256", "passed", "tokens", "start", "swimlane", "pmu"))])]
    c4_expected = dict(projection_units=40, aic_launch_blocks=20, steps_per_lane=2, repeated_physical_input_rows=48,
                       projection_submits_per_chunk=1, candidate_observed_execution_rows="unavailable",
                       forty_aic_cores_proven=False)
    cases.append(dict(
        base="c4_units", raw=c4_raw, extra=c4_extra, required=["q1"], required_current=["q3", "q4"],
        question="For the SELECTED source's first chunk base=0, rows=512, return projection_units, "
        "aic_launch_blocks, steps_per_lane, repeated_physical_input_rows (overlap between adjacent row tiles, "
        "count row positions once, not multiplied by N/weight), and projection_submits_per_chunk. Use unavailable "
        "when selected evidence is absent. Return candidate_observed_execution_rows (integer or unavailable) and "
        "forty_aic_cores_proven (boolean). Distinguish work units, logical submits, launch blocks, actual execution rows, "
        "and the separate AIV pool; source calculation does not measure cores or memory traffic.",
        expected=c4_expected, expected_d0={key: (False if key == "forty_aic_cores_proven" else "unavailable") for key in c4_expected},
        derived=relation(dict(inputs=["q3.0", "q3.1", "q4.0", "q4.1"],
            calculations="OUT_DIM=2*512=1024; ceil(512/112)=5 row tiles; units=5*(1024/256)*2=40; ceil(40/20)=2 steps per lane, unit=lane+step*20. Row starts [0,112,224,336,400] give 48 overlapping input row positions.",
            launch="One AIC submit sets block_num=20. The separate AIV submit sets block_num=40. Source work units do not equal AIC cores or captured physical execution rows.",
            boundary="The selected manifest has swimlane=0,pmu=0. Actual selected physical execution-row count/core assignments and measured redundant memory traffic are unavailable."))))
    pool_raw = [group([bounded(inputs, "pool_source", 215, 233), hints(inputs, "pool_hint", limit=2),
                       bounded(inputs, "spaces", 339, 361), dict(rec="PACKET_SCOPE", evidence="derived",
                       not_supplied_in_historical_groups=["generated placement of the identity multiply", "cross-pool capacity-relief measurement"])])]
    pool_extra = [group([bounded(inputs, "pool_kernel", 616, 637)])]
    pool_expected = dict(warning_space="Right", warning_hardware_pool="L0B", generated_identity_space="Vec",
                         generated_identity_hardware_pool="UB", shrinking_identity_proves_warning_relief=False,
                         identity_removal_synchronization_safety_proven=False)
    cases.append(dict(
        base="swa_pools", raw=pool_raw, extra=pool_extra, required=["q1"], required_current=["q1", "q2"],
        question="Relate the PH-MR-001 storage space to the generated identity multiply. Return warning_space, "
        "warning_hardware_pool, generated_identity_space and generated_identity_hardware_pool "
        "(use unavailable without generated evidence). Hardware pool names: L0A, L0B, L0C, L1 or UB. "
        "Return shrinking_identity_proves_warning_relief and identity_removal_synchronization_safety_proven "
        "(booleans). Separate observed placement from an unmeasured cross-pool relief or synchronization claim.",
        expected=pool_expected, expected_d0=pool_expected | dict(generated_identity_space="unavailable", generated_identity_hardware_pool="unavailable"),
        derived=relation(dict(inputs=["q1.0", "q1.1", "q1.2", "q2.0"],
            edges=["PH-MR-001 -> Right -> L0B", "generated TMULS(v57,v56,v23) -> TileType::Vec operands -> unified buffer (UB)"],
            boundary="Distinct storage pools: no supplied capacity/liveness experiment proves that shrinking the Vec identity operation relieves Right occupancy. This placement evidence also does not prove that removing a mixed-engine handoff preserves synchronization."))))
    return cases


def numbered(groups: list[dict]) -> list[dict]:
    result = copy.deepcopy(groups)
    for i, query in enumerate(result, 1):
        query["id"] = f"q{i}"
        for j, fact in enumerate(query["facts"]):
            fact["id"] = f"q{i}.{j}"
    return result


def prepare(output: Path) -> dict:
    if output.exists():
        raise ValueError("refusing to overwrite a corpus directory")
    inputs = load_inputs()
    cases, mapping, payloads = [], [], {}
    for index, spec in enumerate(definitions(inputs)):
        rotation = ARMS[index % 3:] + ARMS[:index % 3]
        for arm in rotation:
            case_id = f"case{len(cases) + 1:02d}"
            groups = spec["raw"] + ([] if arm == "D0" else spec["extra"])
            if arm == "R":
                groups = groups + [spec["derived"]]
            packet = dict(kind="offline_artifact_evidence", scope="Only supplied artifacts. Fingerprints identify files; source/build association is limited to supplied producer records, not compiler attestation. Missing evidence is not a measurement.", queries=numbered(groups))
            expected = spec.get("expected_d0", spec["expected"]) if arm == "D0" else spec["expected"]
            cases.append(dict(id=case_id, packet=case_id + ".json", question=spec["question"], expected=expected,
                              required_queries=spec["required"] if arm == "D0" else spec["required_current"]))
            mapping.append(dict(id=case_id, base_case=spec["base"], arm=arm, arm_order=list(rotation),
                                identical_raw_negative_control=not spec["extra"]))
            payloads[case_id + ".json"] = packet
    # Detect changing inputs before publishing any corpus file.
    for name, (relative, expected) in FILES.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"input changed during export: {relative}")
    reviewer = dict(version=1, cases=mapping, input_sha256={relative: expected for relative, expected in FILES.values()},
                    generator_sha256=digest(Path(__file__)), runner="run_offline.py (unchanged)",
                    cases_count=len(cases), base_cases=6, replicates=3, planned_calls=54, conditions=["facts"],
                    interpretation="D1 adds raw evidence where available; R preserves D1 raw groups exactly and adds derived relations. Not a pure formatting test. Contract D0/D1 are identical negative controls; metadata hashes already suffice for a collision negative control.",
                    scoring_limit="Typed-field and minimum citation coverage checks do not establish semantic entailment or operator speedup; independent response review remains required.",
                    calibration_limit="Questions explicitly scaffold the relevant distinctions, creating a possible ceiling effect. This bounded evidence-reading calibration does not establish improvement in real optimization decision quality or device performance.")
    output.mkdir(parents=True, exist_ok=False)
    for name, obj in payloads.items():
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    (output / "cases.json").write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=2) + "\n")
    (output / "reviewer.json").write_text(json.dumps(reviewer, ensure_ascii=False, indent=2) + "\n")
    return reviewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))
