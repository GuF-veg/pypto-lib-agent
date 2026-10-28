# pfdb upgrade and release audit

Audit date: 2026-10-21. This is a read-only M5 compatibility audit against the
installed, pinned runtime and existing captures. It does not certify an unseen
future runtime, an optimization gain, or the complete development plan.

## Version boundary

The release work tree reports pfdb **0.5.0**, DB schema **8**, MCP tool schema
**4**. Capture analysis remains adapter `pfdb/0.4.0` / semantics `runtime/1`.
The package version changed independently of these persisted analysis identities.
The changes under review use existing schema-8 columns; no schema-9 migration is
needed for transport recovery, metric definitions, level-1 unavailable wall
timing, scope headers or retention of compact trial benchmark results.

The installed upstream anchors match the pre-experiment toolchain baseline:

| Component | Revision | Tracked binary-diff SHA-256 |
| --- | --- | --- |
| PyPTO | `ee49fceaa3290cf90257a22eab867c04245b499d` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| runtime | `6e383fc57c12a8bd4dce045289b8b4f5d55a15f1` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Both tracked diffs are empty. No checkout, submodule, compiler pin, runtime
source or device acquisition was changed in this audit. Source-of-truth reuse
and compatibility retirement remain in the
[integration ledger](../../../docs/debug-and-tune/pfdb-runtime-integration.md).

## Executed contract checks

All commands ran in conda `pypto`. Logs, JUnit results and assertion summaries
are under ignored `build_output/pfdb_development/upgrade-audit/`.

| Check | Result | Coverage and limits |
| --- | --- | --- |
| `test_schema_migrations.py`, `test_runtime_integration.py`, `test_ingest_modalities.py` | **40 passed** | Migration order/idempotency/rollback; official scheduler/dependency parity; argument decoding; host loss/cold/warm; scope/incore; copy/link/export/cache; optional isolation; stale and failed-refresh preservation; changed-upstream guard |
| Release-specific metric/lifecycle regressions | **4 passed** | Lower-order-statistic median definition and level-1 wall unavailability; compact completed-trial benchmark retention; explicit partial-summary separation; transactional rollback on failed prune |
| Real SWA agent capture, `test_ingest_real_capture.py` | **3 passed, 4 skipped** | Converter row timing, upstream CP parity and applicable stall decomposition; four Qwen-specific constant checks skipped |
| Real SWA manual capture, same suite | **3 passed, 4 skipped** | Same parity scope; same four Qwen-specific checks skipped |
| Explicit schema-7 backup/copy migration | Passed assertions | Source stays schema 7 and hash unchanged; working copy becomes schema 8; notes/bench/CP observation/baseline/trial preserved; CP explicitly stale; refresh without source refused |
| Explicit CP row-width/epoch matrix | Passed assertions | Widths 5/6/7 preserve official raw result; widths 4/8 refused; multiple epochs and mixed epoch-bearing/non-epoch rows refused; input bytes unchanged |
| Two real edgeless holdouts | Passed assertions | RMSNorm and matmul saved CP payloads equal fresh official-adapter results; static/observed queries available; raw hashes unchanged |
| A0 archived API vs current API | Passed assertions | Existing facts identical for five queries and existing LONG record; expected additive metric-definition fact; name-filter bug reproduced before/fixed after |

The CPU suite emits the pre-existing Pydantic warning that `AttachToolParams.copy`
shadows a `BaseModel` member. There were no test failures. Counts above describe
executed test invocations, not independent agent or device samples. The eight
skip occurrences are the same four anchor-specific tests run against two different
captures; they are not validated by the successful parity checks.

The focused test commands are:

```bash
python -m pytest profile_db/tests/test_schema_migrations.py \
  profile_db/tests/test_runtime_integration.py \
  profile_db/tests/test_ingest_modalities.py -q -ra

PFDB_REAL_CAPTURE_DIR="$PWD/build_output/pfdb_development/swa_agent_capture/dfx_outputs" \
  python -m pytest profile_db/tests/test_ingest_real_capture.py -q -ra

PFDB_REAL_CAPTURE_DIR="$PWD/build_output/pfdb_development/swa_manual_capture/dfx_outputs" \
  python -m pytest profile_db/tests/test_ingest_real_capture.py -q -ra
```

## Guarded runtime compatibility

`CP-ROW-001` is still needed at the audited runtime. Its exact
`build_graph` source SHA-256 remains
`a6ecb2590456468f3b40ee0245fb5c14122d38db9d7ee6b6cd7232f51f9f0223`.
Five- and seven-column fixtures use the guarded private normalization; the
six-column fixture does not. All three match the reference official raw path
result. Unsupported row widths and execution-epoch mixtures fail explicitly.

The existing changed-function test replaces the function in its own process,
confirms that the old workaround is **not applied** to an unknown function hash,
and checks raw-byte preservation after failure. This exercises safe refusal;
it is a simulation of interface drift, not validation against a new upstream
release. Optional artifact/parser failure remains isolated from successful
critical-path and scheduler snapshots. Failed refresh retains the previously
successful snapshot, including the old-version case, rather than overwriting it
with unavailable data.

No compatibility was retired or broadened merely to complete this audit.

The standalone audit outputs are `baseline-invariants.json`,
`migration-copy.json`, `width-epoch-audit.json` and `capture-parity.json`. They
record the exact comparison outcomes independently of the test logs; the
archive-based API probe and its two responses are preserved in the same ignored
audit directory. The old schema-7 original and migrated working copy are also
retained there for inspection.

## Existing database and API preservation

All three original study databases match their A0 SHA-256 values before and
after the audit:

| Database | SHA-256 |
| --- | --- |
| `.pfdb/compressor_ratio4_gap.duckdb` | `400b421b188ebdeb22eaf8d88b941b3c7067dc213c879a3e3a90deea177d9f2e` |
| `.pfdb/swa-a2.duckdb` | `ad8d9b1eedd6cfe5d0fad9d9500d68ade6718d227fdadd6cbb04f6617bddba1b` |
| `.pfdb/swa-gap-20261009.duckdb` | `0a9cc33d23d5cd747e2df19007da14773770bf985be4480397f80e0d2b804a6e` |

The pre-change A0 `profile_db` archive was unpacked into an audit-owned directory,
then imported in a fresh process with an explicit source path. No Git checkout
or installed package was replaced. Both APIs opened the original compressor
database read-only. Exact fact equality holds for `overview` (4 records), `bench`
(101), `rows` (64), `critical_path` (3) and `perf_hints` (26). The existing
level-4 `LONG` record is unchanged; current `why_long` appends a
`METRIC_DEFINITION` record. The old task-name filter raises its known keyword
collision, while the current API returns the intended task.

The audit also rechecked all five originally registered files in each fresh SWA
capture, and all five capture files in each holdout. Their hashes match the
earlier manifests. No profiling results were silently recomputed in the original
databases, and no original benchmark, verdict or unknown validation was upgraded
to a proof.

## Release and rollback limits

- Schema-8 readers can retain the same observations; the A0 compatibility check
  above covers representative reads, not every old writer or destructive lifecycle
  operation. Keep a database backup before applying lifecycle cleanup or reverting
  to an older package.
- Schema-7 migration was rehearsed on an audit-owned copy. Queries should not be
  used as an implicit refresh; migrated analysis stays stale until explicit
  reanalysis from identified artifacts succeeds.
- Level-1 wall timing is now explicitly unavailable in relevant query output;
  historical persisted timing rows were not rewritten. Consumers must tolerate
  unavailable/additive facts rather than infer a wall measurement from endpoints
  that were not collected.
- The new package version does not change runtime analysis identity, so unchanged
  official snapshots need no blanket refresh merely because pfdb was upgraded.
- An actual newer runtime was unavailable and deliberately not installed. Reused
  functions, transitive helpers and format changes must be audited at the next
  real pin update; passing this guard matrix does not establish future parity.

This audit did not run paid models, collect new device data, validate general
optimization benefit, publish a package or create a commit.
