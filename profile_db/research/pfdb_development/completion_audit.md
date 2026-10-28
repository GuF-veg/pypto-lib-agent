# Independent audit of M0–M5 completion

Initial audit and closure date: 2026-10-21. **The registered bounded pilot and
independent fixture confirmation are now complete.** The closure below supersedes
the historical pre-confirmation snapshot, which is retained separately. The
closure verification read existing source, manifests, protocols and results and recomputed
statistics from saved samples; it launched no model, NPU, compilation or
profiling collection, reran no hardware/tests, and opened no study database.

The acceptance authority is the [full development plan](../../../docs/debug-and-tune/pfdb-development-plan.md),
especially sections 7–11 and 12–14. Completing only its suggested first round
(M0/M1 and one E1 prototype) would not complete that plan. Conversely, sections
1, 11 and 12.1 explicitly permit retaining capabilities for insufficient
evidence and do not require implementing every proposed enhancement.

## Closure evidence: 2026-10-21

The completed pilot is `build_output/pfdb_development/optimization-pilot-v1/`;
its confirmation is `build_output/pfdb_development/optimization-confirmation-v1/`.
The latter's `results.json`, `measurements.json`, `jobs.json`, `sessions.json`
and frozen `protocol.json` were independently cross-checked:

- All **24 sessions / 12 matched pairs**, four starting implementations and
  **44 model attempts** are present. The [public decision review](online_decision_review.md)
  now covers all 24 sessions; its pending-confirmation wording describes that
  search-only review snapshot, not the final confirmation state.
- There are **11 unique source/workload groups**. Each canonical group identity,
  source file and recorded source origin matches its saved hash. All session
  selected-source hashes match their pilot incumbents; original baselines are
  bound to their own starting-source hashes. Shared confirmations remain shared,
  not new independent measurements for each session.
- All **11 held-out validations and 33 benchmark strata passed**. Saved log
  hashes and PASS markers match the records; raw headline parsing reproduces
  every stored stratum. Each contains 100 finite positive samples with warmup
  5, one rank and matching headline aggregation: **3,300 samples total**.
- Independent arithmetic reproduced the pooled medians, every used 2,000-draw
  within-stratum bootstrap interval, the frozen acceptance gate, final-job and
  fallback choices, and all 12 paired log effects. There are **24 valid final
  sessions, zero missing final measurements, 12 fixture-confirmed win sessions
  and one fallback**. These are session outcomes with shared implementations,
  not 12 unique optimized kernels or a general-domain correctness certificate.
- The fallback is `swa_manual-1-semantic_contract`: baseline median **199.1 µs**,
  selected median **200.25 µs**, baseline/candidate ratio interval
  **[0.9925187032, 0.9957510888]**. The selected candidate is not accepted; the
  independently confirmed original baseline is retained.
- The preserved stable release log
  `build_output/pfdb_development/final-stable-tests.log` reports **378 passed,
  11 skipped**, with the existing Pydantic `copy` warning. This audit read that
  log; it did not rerun tests. The earlier inventory-update race and its failed
  log remain historical evidence, superseded by the stable run.
- The main executor's final invariant records report unchanged SHA identities
  for three original databases, four canonical operator sources and four
  original replay source/build trees; both PyPTO/runtime revisions and empty
  tracked diffs also remain unchanged. The inspected capture-integrity record
  verifies **35 original compressor artifacts and 10 fresh SWA artifacts**.
  These are saved final checks, separate from this audit's independent
  source-group/sample arithmetic verification.
- The [public optimization summary](optimization_summary.json) contains all 24
  curated session rows and costs. Every original outcome field and selected/final
  source SHA matches the raw confirmation; all seven raw-source digest references
  match their files. The final count is **7 treatment / 5 baseline fixture-win
  sessions**. This count is descriptive and does not establish a success-rate
  improvement.

The treatment-over-baseline geometric speedup ratios recomputed from the final
incumbents are:

| Fixed starting implementation | Three-pair geometric ratio |
| --- | ---: |
| C4 agent | 1.0132433233 |
| C4 manual | 0.9297688392 |
| SWA agent | 1.0364606417 |
| SWA manual | 1.0000000000 |
| Four starts, equal weight | **0.9940549405** |

The stored aggregate is `0.9940549405407888`; independent recomputation differs
only by floating-point roundoff. This is a descriptive result for the registered
fixed tasks. It **does not establish treatment superiority, statistical
equivalence or population-wide agent optimization improvement**. Benchmark
intervals are conditional on three fixed processes; reused confirmations and
three session repetitions do not create an agent-population confidence interval.

### Conditional scan result remains conditional

The C4 manual baseline replicate 1 selects source SHA
`6fded29273bcc107963a853cbcdefacb62d9e299866928388c55df891921db2e`.
It passes both declared fixtures and confirms **177.55 µs / 1.3804562095×**
relative to its original baseline on the registered search workload. Its scan
range uses `position_ids[tile_base]` to infer physical token indices, as the
[public review](online_decision_review.md) section 4 explains. The search uses
512 tokens/start 0 and held-out validation uses 512 tokens/start 3; the original
builder's `init_position_ids` uses `arange` for both. Moving the start preserves
the continuity assumption and does not test arbitrary position/mapping domains.

Thus the numerical result belongs in the **fixture-confirmed** account. It is
not evidence that the specialization is generally legal, nor that the semantic
condition is worse because another condition discovered a universally valid
algorithm. The common legal domain must be established before any such claim;
do not silently relabel the accepted fixture gate as full semantic validation.

### M0–M5 disposition after closure

| Stage | Dated disposition |
| --- | --- |
| M0/M1 | Baseline, inventories, isolated packets and public access repairs have evidence; A0/A1/current production identities remain distinct. |
| M2 | Selected exploratory screens are complete. Existing facts/default directory are retained for insufficient evidence; the unrun grouped-discovery and component studies remain explicitly unassessed, not successful experiments. |
| M3 | Registered 24-session search, all-24 public review and independent fixture confirmation are complete. Broader input domains, adaptive pfdb use and cross-operator optimization generalization remain unproved. |
| M4 | Selected correctness/interpretation/retention changes have deterministic and release evidence. No general source/IR system, unsafe PMU association, wholesale directory change or evidence deletion is justified by this result. |
| M5 | Stable tests, final original-database/source/build/upstream invariants, capture integrity, pinned-runtime/old-database compatibility rehearsal and the reconciled public result/decision record are evidenced. This audit authorizes no package publication and makes no unseen-runtime claim. |

The [results](../../../docs/debug-and-tune/pfdb-development-results.md) now supply
per-question D0/D1 information recovery and online session-level usage/cost, and
explicitly disclose: tool-free fixed initial evidence rather than adaptive
queries; no fresh candidate profiling/hints fed back to the model; two candidate
slots frozen before execution; provider seed/revision limits; incomplete blind
free-form review; unrun directory/memory/component contrasts; and inability to
separate pure queue time. Those deviations limit the claim, rather than being
retroactively called completed comparisons. A new default or general benefit
claim still requires its own declared gate and appropriate confirmation.

The previously pending closure items for the **selected bounded study and
correctness release** therefore have evidence. Unrun experiments remain
unassessed/deferred and cannot be relabeled as passed acceptance gates. No
generic agent-benefit, new-default or evidence-deletion claim follows from
closing the execution and reporting work.

Saved closure anchors (SHA-256):

| File | Digest |
| --- | --- |
| Confirmation `results.json` | `410737333e61c839a810bf0aed368dd3944ab708a52da1b130378972dbe0f90c` |
| Confirmation `measurements.json` | `d998c317907ee60151e32042e12b2ce7d917084275c9f82c85a0f43a9519f1de` |
| Confirmation `protocol.json` | `1ab5d72300c82b1ed43413db86d9355b3da3f9b347a2231f8b5a553c48bf9e08` |
| Stable release test log | `e4d0d2d81c198da0f871bdf4d6a273d2f45f5f0b120e8b54abbceb73f463ebac` |
| Final invariant record | `992f0beabd4f8cbb1b54e608ec2e46d1423818b24e767be2386d2aeddb633d67` |
| Final capture-integrity record | `400a6cf7bb5c2518df1d80d5d06968f9e54024cd4b8f435cdaa5ab488e30d924` |
| Public optimization summary | `3d0c116d4431cf226e6aedf0a634de2ab15f8f329f1baa59abfbbf1fa5dae443` |

## Historical pre-confirmation snapshot: superseded status

The following assessment and closure checklist were written while 17 sessions
were available and confirmation had not begun. Their pending statuses are
historical; the dated closure above records what was subsequently completed.

### Stage assessment at that snapshot

| Stage | Evidence delivered | Assessment and remaining acceptance |
| --- | --- | --- |
| M0: baseline and inventory | A0 archive/failure records and original hashes; client Code Mode investigation; frozen ten-question corpus and [registration](study_registration.json); [41 decisions](interface_decisions.csv); [92 capability/view/channel modes](capability_inventory.csv), with schema-driven generator and source/permission/test references | Substantial baseline deliverable exists. Preserve the distinction between frozen A0, transport-only A1 and later repaired production output; a package version alone does not identify an experimental packet. Ignored source artifacts remain necessary for reproduction. |
| M1: reliable access | Public name selector repair; strict positive MCP response budgets; full retry across API/CLI/MCP; read-only lifecycle recovery; recorded source/hash preservation and deterministic tests | Deterministic repair is evidenced, independently of any agent speedup. New reads replace old truncated responses; they are not persistent cursors. The final stable-tree regression remains a separate release check. |
| M2: exploratory prototypes | Ten tasks: D0 facts/columns 60 attempts and per-task D1 30; one-question interactive description contrast, six attempts; declared negative controls and source/identity additions | Selected exploratory prototypes ran. They establish neither a superior directory nor statistical noninferiority. The planned ten-question grouped-discovery comparison was not completed; label it unassessed/deferred, not a successful layout experiment. No replacement default is justified. |
| M3: counterexamples and closed loop | Five synthetic positive capabilities; two real fixed-structure holdouts; separately registered metric-definition follow-up; two additional legal validation fixtures; [public failed-candidate review](online_failure_review.md) and [partial decision review](online_decision_review.md) | **Pending.** At inspection, 17 of 24 pilot sessions had result files; this is a moving snapshot. Complete the registered pilot and independent confirmation, then account for all sessions and applicable public decisions. Holdouts currently test interpretation, not optimization generalization. |
| M4: selected implementation | Access/edgeless-ingest repairs; scoped descriptions; timing definitions and shared level-1 mask; complete/unknown family timing; coherent trial summary retention, bound-run identity and referenced-ID preservation | Concrete correctness work is evidenced by [implementation review](implementation_review.md). It does not require a performance superiority result. No tool regrouping, general source/IR engine or unsafe PMU join was silently adopted. Final documents/tests must match the selected final diff. |
| M5: release and review | [Pinned-runtime compatibility audit](upgrade_audit.md), schema-7 copy migration, A0 representative API parity, raw-width/epoch/changed-upstream guards, real capture parity, backup/rollback guidance and [single runtime ledger](../../../docs/debug-and-tune/pfdb-runtime-integration.md) | Compatibility rehearsal at the available pin is evidenced; an unseen new runtime is expressly unassessed. Final release readiness is **pending** stable-tree checks and final result/decision reconciliation. Publishing a package or tag is not implied by this audit. |

The [study summary](study_summary.json) contains 150 main static attempts, not
all calibration, discovery, acquisition or online work. Exact/type correctness
and citation ID/coverage are distinct from entailment, legal transformations
and useful experiments. The distinction is also explicit in
[corpus notes](decision_corpus_notes.md) and `run_offline.py::score`.

### Required closure work identified at that snapshot

1. **Finish and freeze the 24-session pilot.** Preserve the registered two
   candidate slots, serial/balanced execution, failures and original diagnostics.
   Produce the aggregate `results.json` with all 12 matched pairs, invariant
   checks and exact source/protocol identities. Do not replace slow/failed
   sessions or use the current partial sample as the treatment estimate.
   `confirm_optimization.py::prepare` correctly refuses an incomplete pilot.

2. **Execute independent confirmation before reporting optimization results.**
   Confirm the original baselines as well as selected incumbents, validate the
   held-out legal fixture and collect three fresh 100/5 raw strata per unique
   implementation/workload through the external executor. Retain shared-job
   identities, failed/missing measurements and fallback records. Use the frozen
   median estimator and gate; current pfdb mean bootstrap cannot substitute.
   Its within-stratum interval remains conditional on the observed processes,
   not an interval for agent-population improvement.

3. **Complete the final evidence account.** Report all four starting points,
   all pairs, independent validation, confirmed incumbent latency, valid
   improvement fraction, fallbacks/missingness, failure categories and resource
   cost. Record unsuccessful time-to-improvement as censored at the budget;
   do not average only successful sessions. Static usage is already preserved;
   add online session-level usage and external wall costs once available,
   without double counting shared confirmations or claiming an unmeasured
   queue/compute split. Add a compact per-question D0/D1 information-recovery
   account: correct baseline abstention is not an accuracy failure.

4. **Close the interpretation review.** The current public decision review
   deliberately covers only the first 16 sessions. Reconcile its scope with the
   complete pilot; inspect applicable numerical, memory-space, work-count,
   synchronization and evidence-entailment claims. A successful timing is not
   proof of the proposed mechanism. Existing automated `rubric_passed` flags
   must retain their mechanical scope. If independent/blind free-form review
   was not performed, disclose that deviation rather than manufacture a blind
   assessment after reviewers have seen condition labels and outcomes.

5. **Freeze the final disposition and release checks.** Update the results,
   decision ledger and review notes from the final evidence, then run the
   complete CPU/repository checks on the stable final tree and preserve their
   logs, skips and versions. The first final-suite run raced an inventory
   edit: 377 other tests passed and inventory snapshot equality failed. This
   audit does **not** claim a final full-suite pass; the main executor will
   rerun it once stable. Keep that initial failure log as part of the record.

The existing results text says both pilot and confirmation are still executing;
at this audit confirmation had not started. Change that status to distinguish
the running pilot from pending confirmation. The final report may legitimately
find no supported default change or no confirmed treatment advantage. That
concludes the bounded experiment with insufficient evidence; it does not pass
a superiority/noninferiority gate or prove that omitted information is useless.

## Legitimate deferred work, with explicit limits

| Candidate | Why it is not automatically required implementation | What would make further work necessary |
| --- | --- | --- |
| Grouped/three-entry directory, summary-first output, universal columns | No selected replacement has met the declared quality/cost gate; opposite compression examples and Code Mode change the premise | A proposal to promote/hide/deprecate an entry requires the declared discovery/quality experiment, complete evidence recovery and low-frequency positive cases |
| Component ablations and the complete D0/D1 × P0/P1 grid | No interaction claim is made; per-task D1 differs by question; online `semantic_contract` is explicitly a combined definition/validation condition | Attributing an online gain to one component, deleting evidence or promoting a claimed beneficial combination requires the relevant positive/removal controls and a frozen confirmation protocol |
| Structured workload gate, independent PMU association, broad source/lowering/work indexes, semantic stage mapping | Candidate work packages were conditional on evidence, not unconditional feature commitments; source/contract packets can test need without schema expansion | Implement the selected relation only with identity/provenance and a demonstrated failure/use case; never match PMU by task ID alone or fabricate fused-stage latency |
| Online memory treatment, general DSE or transformation engine | Only the small retention/identity defects and a synthetic history interpretation scenario were selected; a new search system was not justified | A separate claim about optimization benefits from structured memory needs a controlled shared-history experiment, not merely the retention fix |
| Arbitrary position/mapping inputs, visual-agent utility, optimization on extra operators | Two legal validation fixtures and two interpretation holdouts are explicitly bounded; retained low-frequency capabilities need not all be optimized now | Broader correctness/generalization claims or removal/default-hiding decisions require the corresponding real positive and counterexample coverage |
| An actual newer runtime or retiring CP-ROW-001 | The available pin still needs the exact-hash compatibility; the plan forbids arbitrary toolchain updates just to complete a checklist | At a real pin update, inspect function/transitive changes, run parity/guard regressions and retire compatibility only when the upstream fix passes the saved-capture criteria |

A later agent-quality confirmation is necessary before claiming broad treatment
benefit, not merely because a 24-session exploratory pilot exists. The current
held-out fixture and three benchmark strata confirm candidate validity/timing;
they do not create unseen agent sessions or estimate agent-population variance.
No production default should be promoted from an unregistered post-hoc threshold.

## Inventory correction resolved during this audit

The inventory's manual task note incorrectly called level-1 lifecycle exposure
an unresolved adjacent presentation limit. The shared query mask already fixes
that presentation while preserving raw stored placeholders. Only that note in
`inventory_capabilities.py` and its generated CSV was corrected, as explicitly
requested; the **three focused inventory tests passed** afterwards. The A0 tool
inventory and 41 decisions were not overwritten. This resolves the stale claim;
it is not an additional runtime or profiling change.

At that snapshot, full-plan closure still required the experimental and stable
release records now addressed above. The unchanged rule is that completion does
not require turning every unselected hypothesis into a feature, deleting tools,
inventing an upstream upgrade, or reporting an optimization improvement that
the final confirmation does not show.
