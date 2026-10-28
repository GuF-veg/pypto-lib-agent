# Focused production correctness review

Review and follow-up date: **2026-10-21**. The latest pass includes the central
level-1 mask and family-comparison completeness correction described below.

Reviewed the current `profile_db/src` diff, its callers and updated regression
tests. Research harness behavior is outside this review. No original database,
capture, upstream source, device or model was changed or invoked.

## Findings and resolution

Two concrete source-identity defects and an adjacent pre-existing family-timing
defect were reproduced with audit-owned synthetic data. All three are now fixed
in the production diff:

| Finding | Reproducer and consequence | Resolution |
| --- | --- | --- |
| P2: retained benchmark could be assigned to a different run | Bind a trial to run 1 (mean 100), finish/prune, then bind it to run 2 (mean 50). The old summary was returned beside run 2. | [bind_trial](../../src/profile_db/lifecycle/trial.py) refuses a different run once bound; a new child trial is the migration path. Binding the same existing run remains idempotent. |
| P2: ingest recycled a still-referenced pruned run ID | Public ingest → trial bind/done → prune all → ingest a distinct capture. `max(run_id)+1` returned 1 again, making the historical locator refer to new evidence without any explicit rebind. | [writer.next_id](../../src/profile_db/ingest/writer.py) reserves the maximum ID across current runs and surviving trial/baseline references. No schema change or invented provenance was added. |
| P2, pre-existing: missing family timing became measured zero or an incomplete total | Two untimed family instances with `busy_us=NULL, wall_us=NULL` produced measured zero deltas through `COALESCE(SUM(...),0)`. One timed plus one untimed task silently produced a partial family total. | [compare](../../src/profile_db/lifecycle/compare.py) requires a non-NULL value for every task of a present family, independently per metric. Incomplete timing deltas are omitted and identified by scoped `unavailable_metrics` entries; task counts and other fully known metrics remain. A truly absent family still represents zero tasks/work. |

The two focused public-API regressions in
[test_lifecycle.py](../../tests/test_lifecycle.py) independently pass (**2 passed**).
The second test uses actual ingest after pruning, rather than manually choosing
the next run ID. Logs are in the ignored upgrade-audit directory. This prevents
new corruption; it cannot reconstruct provenance already lost by an old writer.

The completeness correction is covered by all-NULL, partially timed,
before/after reversal, mixed known/unknown families, JSON rendering and absent
family cases. Known busy-delta ordering is preserved; unknown totals sort after
known totals. It changes neither runtime analysis nor stored observations.

No outstanding newly introduced correctness defect was found in this focused
scope. This is not an audit of every existing subsystem.

## Checked contracts

- [API query selector](../../src/profile_db/api.py): positional selector and
  `query_name=` leave `name` available as a task filter; the legacy
  `query(name="overview", ...)` spelling still works. CLI/MCP callers use a
  positional selector, removing the observed collision.
- [MCP transport](../../src/profile_db/mcp_server.py): query and read-only lifecycle
  budgets are optional strict positive integers, separated from selectors. The
  effective budget is used for rendering as well as query execution. Invalid
  parameters become `QueryError`; writes do not acquire a retry-budget field.
  Existing transport tests cover real stdio recovery, CLI/API byte parity,
  invalid budgets and long trial-history recovery.
- [Edgeless input](../../src/profile_db/ingest/deps.py): an empty list is accepted,
  while missing/non-list edges still fail. Nonempty-edge validation is unchanged.
  Level-1/4 tests and the two independent-task real holdouts cover the boundary.
- [why_long](../../src/profile_db/query/handlers_z4.py): definitions match ingest
  envelopes; row lifecycle duration is distinct from pipe activity; the existing
  lower-middle order statistic is named explicitly rather than silently changed.
  Level-1 wall timing is omitted and untimed tasks do not receive a fabricated
  family rank. The corresponding tests cover these cases.
- [Comparison header](../../src/profile_db/lifecycle/compare.py): new fields identify
  the capture/profile gate and explicitly state that workload/validation were
  not checked. They do not certify candidate legality. Level-1 family comparison
  suppresses `family_wall_us`; absent task timing suppresses only the affected
  family/metric delta, with explicit unavailability. Existing complete deltas
  retain their values.
- [Shared registry](../../src/profile_db/query/registry.py): the previously open
  level-1 presentation issue is now **resolved**. TASK facts from task/tasks/region,
  ROW facts from rows/core, and PATH facts omit unobserved dispatch/receive/FIN
  endpoints and associated wall/gap values. An unavailable evidence record
  identifies the missing lifecycle stream. The public API regression checks six
  views and confirms the original stored values remain unchanged; the level-4
  golden QA continues to preserve real zero timestamps.
- [Trial summary](../../src/profile_db/lifecycle/trial.py) and
  [prune](../../src/profile_db/lifecycle/prune.py): summaries come entirely from
  one source. A partial explicit summary does not borrow missing fields from
  the bound run. Automatic retention occurs inside the prune transaction, before
  deletion; the existing injected-failure test verifies rollback. Copy-file
  cleanup remains post-commit, as documented before this change.

## Compatibility and remaining boundaries

Package metadata and server initialization identify **0.5.0**. DB schema **8**
and MCP tool schema **4** remain consistent with unchanged tables and tool
names/selectors plus optional budget arguments. Adapter `pfdb/0.4.0` /
`runtime/1` still identifies the unchanged persisted official-analysis format;
it is not intended to equal the package version. Rejecting trial reassignment
is an explicit 0.5 behavior correction and needs the documented child-trial
migration path.

The earlier note about exposed level-1 TASK/ROW lifecycle zeros is superseded
by the central-mask resolution above. Masking remains a presentation operation;
raw storage retains historical placeholders. Official end-based path work and
real row start/end observations are not removed with unavailable FIN timing.

The run-ID correction reserves IDs present in surviving run/trial/baseline rows;
it does **not** introduce a database-wide persistent sequence. A synthetic
database with every run pruned and no surviving lifecycle reference can still
allocate ID 1 again. External unregistered references therefore do not acquire
a new permanent identity guarantee. The public guidance correctly limits its
claim to IDs still referenced by lifecycle memory; older identity corruption
cannot be reconstructed automatically.

`unavailable_metrics` retains plain metric names for capture-wide omissions and
uses `{family, metric, unavailable_before, unavailable_after, reason}` objects
for incomplete per-family totals. The handbook documents this response shape.
Consumers must distinguish an unavailable total from a measured zero and must
still check workload/validation independently of capture compatibility.

## CPU validation

- Before the missing-timing correction: **96 passed** across query, lifecycle,
  API, new-contract, transport and golden-QA tests.
- After the correction: **49 passed** across lifecycle/query/API/transport;
  the final expanded lifecycle file independently reports **26 passed**.
- Supplementary new-contract/golden-QA/dogfood checks: **58 passed**; the
  profiling-semantics comparison gate: **1 passed, 29 deselected**. Ruff and the
  public documentation/link checker pass. The final lifecycle rerun after the
  response bookkeeping adjustment again reports **26 passed**.
- No NPU, model, profiling collection, original database/capture or upstream
  source was involved. The only warning in the combined checks is the existing
  Pydantic `AttachToolParams.copy` shadowing warning.

This review does not establish optimization benefit, arbitrary future-runtime
compatibility or absence of all pre-existing defects. It does not require fixing
the known unrelated Pydantic `copy` warning or adding new interface features.
