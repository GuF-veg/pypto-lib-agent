# pfdb candidate identity, scope and availability

pfdb 0.6.0 adds candidate evidence to existing trials. It reads a producer's
existing manifest and artifacts; it never compiles, runs, validates an operator,
or collects profiling. A compile failure or bench-only candidate needs no
fabricated profile run. The tool count stays unchanged.

## Two existing operations

```bash
pfdb trial register --goal "lower latency" --hypothesis "smaller K tile"
# Use the returned trial ID. The external harness has already written these files.
pfdb attach existing_candidate/candidate.json --trial-id 1
pfdb trial list --trial-id 1 --budget 32768
```

Python uses `db.attach(path, trial_id=1)` and `db.list_trials(trial_id=1)`;
MCP uses `pfdb.attach` and `pfdb.list_trials` with the same parameters.
Attach requires exactly one of `run_id` or `trial_id`. Existing run attachments
and their `copy` option retain their behavior; trial attachments link files and
store the manifest declaration in DuckDB, without a copy mode.

The import returns `source_sha256`, `build_id` and `scope_sha256`. To ask about
an exact candidate, include the expected identities:

```python
identity = db.attach("existing_candidate/candidate.json", trial_id=1)
result = db.list_trials(
    trial_id=1,
    source_sha256=identity["source_sha256"],
    scope_sha256=identity["scope_sha256"],
)
```

The two guards are optional, require a selected trial, and use lowercase full
SHA-256 values. A mismatch returns unavailable identity facts, without another
candidate's scope or benchmark. Source/build files changed or missing after
import similarly suppress current scope and benchmark output. Missing optional
files produce partial availability without invalidating healthy independent
artifacts. Reads recheck file digests, not only names, timestamps or signatures.

Unfiltered `trial list` remains compact: recorded identities have
`candidate_state=unchecked`. Select one trial to check its files and see scope
and availability. A legacy trial without a manifest has no inferred candidate;
its selected candidate view reports unavailable. Read-only schema-8 databases
remain readable; opening for writing applies schema 9 transactionally.

## Read a registered diagnostic

`list_trials` keeps its default summary behavior. Select `view="diagnostics"`
to read an existing diagnostic for one exact source; both `trial_id` and the
expected `source_sha256` are required. `scope_sha256` remains an optional guard.

```bash
pfdb trial list --trial-id 7 --view diagnostics \
  --source-sha256 <expected-source-sha256> --budget 32768
```

Python and MCP use the same `list_trials` inputs (`view="diagnostics"` and
`budget_bytes` for MCP). No additional tool or run is required for a failed
compile. The view returns checked candidate identity and
`CANDIDATE_DIAGNOSTIC` records, without repeating the full artifact list or
returning a neighboring trial's benchmark.

Content is data declared by the producer, not a pfdb diagnosis or instructions
to execute. The reader verifies the entire file's SHA from the same open file
whose content it reads, refuses escaping paths/nonregular files, and rechecks
source/build files before returning. Missing, changed or mismatched evidence
does not return text. The original manifest file need not remain present.

Each file returns at most its final **16 KiB of source bytes**, with total byte
size, byte offset, and `content_truncated`. UTF-8 decoding replaces invalid or
partial characters and is labeled `decoding=utf8_replace`. Output still obeys
the normal response budget; follow `TRUNCATED`'s budget hint. For an excerpt
outside the retained tail, use the registered file with a file-reading tool;
this view does not claim arbitrary file paging or parse an error's cause.

## Manifest v1

The following is a template: replace digest/revision placeholders with the
producer's actual values. All referenced files must already exist below the
manifest directory. Absolute paths, traversal, escaping symlinks, wrong hashes,
unknown keys and duplicate files within a group are rejected. The manifest is
limited to 8 MiB. Its validated declaration is stored in the database, so the
original JSON may later be removed; referenced source/build/evidence files
remain linked and their loss is reported.
Keep the external manifest with its artifacts when rebuilding a deleted
database matters; trial snapshots survive run pruning, not database deletion.

```json
{
  "version": 1,
  "producer": "external-harness/1",
  "source": {"path": "candidate.py", "sha256": "<source-sha256>"},
  "toolchain": {"pypto": "<revision>", "runtime": "<revision>", "platform": "a2a3"},
  "build_files": [{"path": "kernel.bin", "sha256": "<program-sha256>"}],
  "workload": {"tokens": 512, "position_fixture": "arange(0,512)"},
  "contract": {"authority": null, "assumptions": ["positions satisfy p[j]=p[0]+j"]},
  "validation": {"reported_status": "not_run"},
  "artifacts": {
    "compiled_meta": [{"path": "compiled_meta.json", "sha256": "<metadata-sha256>"}],
    "perf_hints": [{"path": "perf_hints.log", "sha256": "<hints-sha256>"}]
  },
  "requested": {"profile": false, "pmu": false}
}
```

Required fields are `version`, `producer`, `source` and nonempty `workload`.
Build files and toolchain can be absent for failed compilation; supplying build
files requires a nonempty toolchain identity. JSON metadata alone cannot be a
build file. An absent build yields `build_id=null`, `build_state=unavailable`.

Artifact kinds are `records`, `deps`, `name_map`, `compiled_meta`, `perf_hints`,
`memory`, `ir`, `diagnostic`, `bench` and `pmu`, each a list of path/hash pairs.
The three core-capture kinds allow at most one file each. `requested` accepts
optional booleans for `profile`, `perf_hints`, `memory`, `ir`, `diagnostic`,
`bench` and `pmu`; an omitted request remains unknown.
`not_provided` means a request was declared but no file was listed in this
manifest. It does not assert that runtime failed to emit a file elsewhere.

`contract.authority` optionally anchors an existing policy/spec file.
`authority_state=provided` means its bytes are anchored, not that pfdb certified
its author or interpretation. Assumptions are producer declarations. Without
an authority the state is `unresolved`; an initializer, fixture or PASS does
not promote an assumption to a supported general input-domain rule.

Validation defaults to `reported_status=unknown`: no supported result was supplied.
Use explicit `not_run` only when the producer knows comparison did not execute;
an exception, missing report, or changed fixture does not establish that.
Existing manifests with explicit `not_run` remain readable and are not rewritten;
earlier producers also emitted that default, so unbound historical records do
not prove nonexecution. This default correction is available in pfdb 0.8.1.
A producer reporting `passed`
or `failed` must supply four path/hash anchors: `report`, `reference`,
`comparator`, and `fixture`, plus any descriptive `scope` object. A fixture
anchor may be the producer's input-hash manifest; pfdb does not recursively
re-run its data generation or certify its transitive contents. Store fixture
coverage separately from the general input contract.

Recognized `source_sha256` and `reference_sha256` fields and a boolean `passed`
in a JSON validation report must agree with the declaration. A recognized
`source_sha256` in small JSON compiler metadata/diagnostics must also agree.
Other formats stay producer-declared; absence of such a source field reports
`validation_source_binding=unavailable`. All validation/authority claims have
`evidence=unproven`, `validation_binding=producer_declared_not_certified`.
File checks establish byte identity, not mathematical correctness or a PASS
gate, and never turn a caller verdict into a certified result.

## Identity and lifetime

`source_sha256` identifies the primary source bytes. `build_id` fingerprints
that source digest, declared toolchain and sorted declared build-file paths and
digests. A matching `compiled_meta.json` does not establish build equality.
Dependencies and complete program coverage remain the producer's responsibility;
source equality alone does not prove equivalent imports or compiled programs.

`scope_sha256` fingerprints workload, declared contract/authority and validation
scope/reference/comparator/fixture anchors. Validation outcome/report are
excluded, so a result is distinct from its tested scope. These are conservative
fingerprints of declared structure, not a semantic-equivalence test.

One trial accepts one immutable manifest. An exact repeat is idempotent; changed
manifest bytes or another location are refused. Use a child trial for new
source/build/scope, another capture, or later artifacts, even if the source is
unchanged. Trial metadata survives run pruning. No original artifact is updated
or deleted by candidate import.

Binding a candidate trial to a run additionally requires its `records`, `deps`
and `name_map` digests to match the ingested run. This check applies whether the
run or manifest is attached first. It establishes a declared core-capture
association, not certification of arbitrary optional run attachments, workload,
source semantics, benchmark provenance or every generated file.

## Small, explicit output

| Fact | Meaning |
| --- | --- |
| `CANDIDATE` | Source/build/scope IDs; `available`, `partial`, `stale`, `mismatch` or `unavailable`; binding remains producer-declared |
| `CANDIDATE_SCOPE` | Workload, assumptions, authority state, reported validation and tested scope; unproven claims |
| `CANDIDATE_AVAILABILITY` | Per modality: `available`, `empty`, `unavailable`, `not_provided`, `not_requested` or `unknown_request` |
| `CANDIDATE_ARTIFACT` | File kind, relative path, expected digest, and `available`, `empty`, `changed`, `missing`, `outside_root` or `unreadable` |

For a registered modality, missing/changed/unreadable members or an incomplete
required profile set remain `unavailable`. With all registered members intact,
`empty` means **all** members have zero bytes; any nonempty member makes the
aggregate `available`. Per-file states remain separate. In particular, empty
stderr does not hide nonempty stdout.

`available` describes checked files, not successful parsing or an analyzed
profile. The candidate view always marks `observed_execution_rows` unavailable;
input token counts, launch blocks and file presence cannot supply that metric.
Ingest and query an actual matched run for runtime-defined observations. Normal
runtime analysis, timings and cache semantics remain unchanged.

Trial benchmark summaries remain explicitly unchecked for source/workload
binding (`bench_binding=unchecked` is beside the summary in the first TRIAL fact,
so a truncated artifact list cannot hide this boundary). Candidate import does not certify summaries supplied by `attach-bench`
or a legacy run fallback. A caller must keep its independent acceptance gate.
No query emits an automatic adoption verdict.

The normal output budget applies. MCP `budget_bytes`, CLI `--budget`, and API
`format_result` support the existing retry-with-larger-budget recovery. Files
remain references: this view does not dump full source, IR or every raw log.

## Selected compiler context without another tool

An external producer may register selected existing compiler text, such as
`ptoas/enorm_quant.pto`, in `artifacts.diagnostic` with its path/hash anchor.
The same reference can be listed under `artifacts.ir` for availability metadata.
The existing source-guarded diagnostics view reads its bounded text tail; it
does not execute, reinterpret or automatically locate compiler transformations.
Register relevant files for the current question rather than appending every
IR/C++ file to default feedback. Budget truncation and tail limits still apply.

Use a new manifest/child trial when extending an existing handoff. Preserve the
original capture and bind compiler evidence to the source/build/fixture that
actually produced it; a File hash alone does not certify that relationship.
The [native study](pfdb-native-operator-study.md) verifies this existing path on
an MTP cast-lowering failure and records its limited information ablation.
