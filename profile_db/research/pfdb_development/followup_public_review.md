# Follow-up public-response review

Reviewed all 54 supplied responses: 54 present, 0 missing. The first 18 records were preserved when the remaining 36 were appended. All individual judgments were completed before aggregation. Detailed claim excerpts, packet-local evidence, calibration notes, and experiment ratings are in `followup_public_review.json`.

## Evidence support

Identified 11 distinct unsupported established propositions across 11 responses; 43 present responses had none identified. This is an evidence-support judgment, not a blanket correctness or deployment approval. Repeated assertions of one proposition count once within a response.

- Seven responses (`item02`, `item03`, `item18`, `item20`, `item23`, `item51`, `item52`) assert that arbitrary consistent paging breaks the affine scan equivalence. Their own supplied scans retain the same guard and select the same indices under `p[j]=p[0]+j`. Changing positions to non-affine values is a separate premise.
- Three responses (`item05`, `item41`, `item42`) report 512 observed execution rows. Their packets supply `tokens=512`, `swimlane=0`, and `pmu=0`; fixture size does not establish captured execution-row count.
- `item32` says the two source files have different byte sizes. Its packet lists both as 34,840 bytes. The different hashes and different MemoryReuse sizes are supported.

Category frequencies count responses and may overlap; the three execution-row claims each receive both count-error and source/observation-conflation categories.

| Issue category | Responses |
| --- | ---: |
| `unsupported_fact` | 7 |
| `unsupported_causality` | 0 |
| `contract_overgeneralization` | 0 |
| `source_phase_conflation` | 3 |
| `memory_pool_conflation` | 0 |
| `unobserved_overlap_speedup` | 0 |
| `precise_count_error` | 4 |

No established overlap/speedup, cross-pool relief, or general-domain certification claim was flagged beyond the concerns above. Correctly calibrated unavailable answers were accepted: missing selected artifacts were not replaced with historical-source facts, and a proposed test was not counted as an already obtained result. A valid limitation statement can coexist with an unsupported assertion elsewhere in the same response.

## Proposed experiments

Rated 39 adequate, 15 partial, and 0 poor. Adequate proposals target missing provenance, selected-source compiler artifacts, actual execution/overlap observations, explicit retained-versus-removed occupancy comparisons, or separate owner-domain and device obligations. Partial proposals generally omit a needed baseline, synchronization criterion, execution-row accounting, or owner-domain obligation. A useful premise diagnostic remains distinct from policy and device validation; repeating a fixture or contract cannot certify a general domain. Experiment quality does not cancel an unsupported claim.

## Visibility and provenance

Judgments used each response's own question and supplied packet plus the frozen rubric. Both public files were checksum-verified; corresponding packet objects in the second batch were verified identical to the same packet IDs in the first. No facts were transferred between different packet IDs. Treatment labels, mappings, deterministic scores, expected answers, registrations, generators, prior reports, and hidden reasoning were not consulted. No model, network, or device calls were made for this review. Evidence content itself was visible and cannot be fully blinded. These descriptive totals support no treatment-effect conclusion.

- `build_output/pfdb_development/followup-v1/blind/public-batch1.json` (216,201 bytes): SHA-256 `bd7d252347a5903b37af63cbb6a398ec5cd4c071101037b2d2836fb144dbd1f5`.
- `build_output/pfdb_development/followup-v1/blind/public-batch2.json` (232,289 bytes): SHA-256 `a8f372180a9d298dd78db1a82f4ab0580e6180542ab4d595046527ade7b857a9`.
- Frozen `followup_review_rubric.md`: SHA-256 `2a08842708c0ebf6e92b80cf0fa0d62224fffef48c3edb659fb9692787807953`.
