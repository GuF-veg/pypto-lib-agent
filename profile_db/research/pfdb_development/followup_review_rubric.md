# Follow-up public-response evidence review rubric

## Review boundary

Review each item independently using only its question, supplied facts, and public response. Do not consult other packets, repository evidence, prior studies, registrations, generators, answer keys, scores, treatment labels, or mappings. Do not infer intent or private reasoning. Evidence content can reveal distinctions, so this is a review with withheld labels and scores, not full blinding. Freeze this rubric before opening the designated response file; aggregate only after every item has an individual review.

## Claim assessment

Count distinct unsupported factual or causal propositions asserted as established. Split independently checkable propositions; count a repeated proposition once per response and note repetition when relevant. A proposition may receive multiple issue categories but contributes only once to `unsupported_claim_count`. Count a contradicted claim as unsupported and explain the contradiction. Do not count an explicitly conditional hypothesis, proposed measurement, or calibrated statement of uncertainty as an established claim. Simple calculations are supported only when their inputs, units, scope, and definitions follow from the packet.

Record exact response wording or a short identifiable excerpt, the relevant supplied fact or absence, and why the claim exceeds it. Use these categories separately:

- `unsupported_fact`: an asserted fact lacks support in this packet.
- `unsupported_causality`: correlation, eligibility, or chronology is promoted to a causal explanation or proof without discriminating observations.
- `contract_overgeneralization`: a contract, check, validation, or successful example is extended beyond its specified domain. Repeated validation under the same contract cannot certify a general domain.
- `source_phase_conflation`: evidence from different sources, phases, captures, timing domains, or scopes is treated as interchangeable without an explicit bridge.
- `memory_pool_conflation`: distinct memory pools, capacities, allocation scopes, or lifetime quantities are merged without packet support.
- `unobserved_overlap_speedup`: measured overlap, actual concurrency, dispatch behavior, latency improvement, or speedup is claimed without the observations needed by the question. Eligibility or a proposed optimization alone does not establish realization or benefit.
- `precise_count_error`: an exact count, total, ratio, or other precise quantity disagrees with the packet or is asserted without the required accounting.

Use the most specific category; add `unsupported_fact` only when it describes a separate concern. Do not demand evidence irrelevant to the question. Omission alone is not an unsupported claim, but record a material unanswered part separately.

## Unavailable answers and proposed experiments

Mark `calibrated_unavailable` when the response correctly identifies what cannot be concluded and stays within the packet. This can coexist with unsupported claims elsewhere. Requesting missing data can be useful; absence of a numeric answer is not itself a defect.

Assess `next_experiment` by whether its requested observation can resolve the specific missing evidence or distinguish live explanations:

- `adequate`: targets the decisive uncertainty, identifies an observable comparison or needed data, and states enough scope or interpretation to discriminate outcomes. A precise request for the missing evidence can qualify.
- `partial`: relevant and potentially useful, but omits a needed comparison, scope, observable, or link from outcome to conclusion.
- `poor`: absent, irrelevant, circular, assumes the conclusion, or repeats validation inside the same contract while claiming it certifies broader behavior.

Judge experimental quality separately from claim support. A strong experiment does not erase unsupported assertions; a calibrated unavailable response need not invent a conclusion.

## Per-item record and aggregation

Record `item_id`, `response_status` (`present` or `missing`), integer `unsupported_claim_count`, `issue_categories`, `experiment_quality` (`adequate`, `partial`, or `poor`), and short evidence for each concern and the experiment rating. Include `calibrated_unavailable` and any material unanswered requirement as separate annotations.

For an absent response, use `response_status: missing`, zero observed unsupported claims, category `missing_response`, and `experiment_quality: poor` with evidence that no experiment was supplied. Zero observed claims for missing content is not a pass. Keep missing items explicit in all denominators.

After all item reviews, summarize coverage, missing responses, claim counts, category frequencies, and experiment ratings. Do not infer treatment effects or decode withheld mappings. Describe conclusions as evidence-support judgments on the supplied packets, with the stated visibility limits.
