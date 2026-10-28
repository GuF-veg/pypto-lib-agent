# Ten-question decision corpus

`prepare_decision_tasks.py` reads two existing databases and the fresh SWA compiler output. It never compiles, launches hardware, edits a capture, or imports findings from the performance reports. The exporter refuses an existing output directory and checks both database hashes after all read-only queries.

The completed development export is `build_output/pfdb_development/decision-corpus-v2/`. Both `d0/cases.json` and `d1/cases.json` use the existing `run_offline.py` format. Those indexes contain **reviewer-only expected answers**; only their question and referenced packet may reach the model. Tool-free agent isolation remains the runner's responsibility. Raw source files and databases stay unchanged. Exact repository, home, Python-environment and configured PyPTO-root prefixes are replaced with named markers equally in all conditions.

D0 is a fixed saved-query packet. D1 appends the explicitly declared source/contract/identity evidence for that question. D0 is an exact prefix of D1's query list, including order and fact IDs. This is an **information** intervention, not a pure formatting intervention. There is no claim that all added evidence belongs in pfdb or that every question contains a genuine gap.

| Question | Source and scope | What can be scored | What cannot be claimed |
| --- | --- | --- | --- |
| submit_grid | Fresh SWA agent run 1, eight `o_proj_a_quant` tasks; same capture's orchestration | Logical tasks, blocks and physical mixed-engine rows; no merged submit | No measured benefit from changing submission organization |
| lowering_identity | Fresh SWA agent attention task and its generated AIV source | Exact `SCATTER_ROW` form; abstention without generated evidence | No PTO-ISA implementation cost or universal scatter diagnosis |
| completion_boundary | Same SWA consumer, direct incoming edges and unchanged source loop | Distinct projection producers; no license to delete a live edge | No alternate grouped consumer or historical narrowing result |
| realized_pipeline | Same SWA PH-MR-001 hints; request-site source | Requested two vs fitted one; no isolated latency attribution | Existing hints are already sufficient; this is a negative control |
| local_global_ranking | Existing compressor runs 1/2, T=512 fixture observations | Reported median ranking differs from local projection-span ranking | General common-domain numerical equivalence, isolated fusion benefit |
| bounded_scan_legality | Compressor run 1 observation plus current reference selection loop | Current PASS is insufficient; reference selects by position/mapping | A new legal-domain counterexample, or validation of a proposed patch |
| row_span_semantics | Existing compressor run 6, T=8192, second projection task and full physical rows | Envelope formula, wall duration and longest row | Pipe-active time from lifecycle timestamps |
| occupancy_objective | Fresh SWA runs 1/2 with no attached unprofiled benchmark | More occupancy/rows is insufficient; missing objective measurement | A no-benefit grid intervention or a new speedup |
| capture_identity | Existing compressor runs 1/6 plus original external `result_s4_p0.json` identities | Different token counts; task ID alone cannot bind independent PMU | Universal equivalence or safe PMU attachment |
| experiment_memory | Original compressor DB has **zero** trial records; existing verdict setter source | Correct abstention and caller-verdict semantics | History usefulness, repeated-hypothesis avoidance, or prune recovery |

The two original compressor result manifests identify source SHA, T=512 vs T=8192, start=0, level-4 profiling, PMU=0 and a PASS on their recorded fixture. They do not certify the full tensor signature's legal domain. Fresh SWA compilation has T=8, H=64, head dimension 512, window 128 and level-4 rows; its compile metadata does not itself carry a cryptographic source-to-build certificate. Excerpt facts preserve their source SHA and line interval; the corpus does not promote that to independent device validation.

`required_queries` names minimal provenance groups; optional source snippets in negative controls are not forced citations. Factual answers, minimal mechanical citation coverage, citation entailment, open hypothesis quality and recovery of newly available information must be reported separately. Condition-specific `unavailable` is a correct D0 answer; D1's additional answerable field is information recovery, not proof that the baseline model was inaccurate.

Run the small CPU checks with the `pypto` environment active:

```bash
python -m pytest profile_db/research/pfdb_development/test_decision_corpus.py -q
ruff check profile_db/research/pfdb_development/prepare_decision_tasks.py profile_db/research/pfdb_development/test_decision_corpus.py
```

These checks passed locally: 2 tests and lint. The real-export integrity check explicitly skips when the ignored local captures are absent. No model or hardware experiment was launched by this exporter work.
