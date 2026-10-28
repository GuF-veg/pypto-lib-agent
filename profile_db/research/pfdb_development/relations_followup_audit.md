# Offline candidate identity and execution-relation audit

Date: 2026-10-22. Scope: the **work/resource relationship** and **new-candidate
evidence binding** priorities. The separate input-specialization priority is
not expanded here. This audit read existing source, generated files, manifests
and logs; the runtime's official swimlane reader was used to count existing
baseline rows. It changed no production/runtime code, capture, database or frozen
packet and launched no model, compiler or NPU. This document is the only new
deliverable. It proposes at most two real-case packets per assigned priority.

## 1. Experimental distinction

Use the main executor's three declared conditions:

- **D0:** the fixed starting-source/profile facts, without the candidate's new
  compilation/validation artifacts. A question about the candidate's actual
  buffers or physical execution can correctly be answered unavailable.
- **D1:** D0 plus the candidate's raw source, logs, small generated excerpts and
  an explicit source/build/artifact manifest. These are newly supplied raw
  evidence and identity information, not a formatting intervention.
- **R:** exactly the same raw evidence as D1, plus narrowly derived work or
  allocation-use associations with source-span references. R adds deterministic
  derivations/indexing and changes presentation. **It is not a pure format-only
  comparison:** an explicit count or relation previously requiring calculation
  is additional derived information, even though no new raw capture is added.

Keep raw fragments, units, failures and unavailable fields accessible in R.
Do not turn a compiler hint into an optimization instruction or a generated
buffer relation into a claim of measured pipe overlap. The old pilot explicitly
did not feed new candidate hints/IR back to the model; those facts cannot be
called information it previously ignored.

## 2. Evidence roots and identity boundary

The following aliases expand to repository-relative paths only:

| Alias | Path |
| --- | --- |
| B | `build_output/pfdb_development/swa_manual_capture` |
| S | `build_output/pfdb_development/optimization-pilot-v1/swa_manual-0-baseline/attempt-0` |
| C | `build_output/pfdb_development/optimization-pilot-v1/c4_agent-0-semantic_contract/attempt-0` |
| F | `build_output/pfdb_development/optimization-pilot-v1/c4_agent-0-baseline/attempt-1` |
| A | `build_output/pfdb_development/c4_agent_capture` |
| G | `build_output/pfdb_development/swa_agent_capture` |
| GS | `build_output/pfdb_development/optimization-pilot-v1/swa_agent-0-semantic_contract` |

Full SHA-256 anchors are in section 7. Model-visible excerpts must redact the
same repository/environment prefixes in all conditions while retaining original
file hashes and line coordinates. Original files remain unchanged. Neutral case
IDs should not advertise the expected answer.

An artifact group should record, at minimum: candidate/parent source digests,
producer and toolchain identity, workload/validation/golden scope, source and
generated file digests, entry/kernel and line coordinates, collection flags,
process result, validation result and any missing evidence. Build association
is a **producer claim supported by these preserved records**, not an infallible
compiler attestation. Reject mismatches; do not repair them by matching a kernel
name, task number, relative filename or API signature.

There is a concrete signature collision: **B and S have byte-identical
`compiled_meta.json`**, SHA `a818d7044dfea0a1e694286ef88aa534e389d2b97e483fb339b56228a2534ac0`,
despite different source, hints and MemoryReuse output. It describes the same
signature/platform; **its hash alone is not a build identity**.

## 3. Candidate-evidence priority: two packets

### I1. SWA K256→K128: reject stale hints and inspect outer-chunk association

**Source and validation.** Baseline source is
`models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py`, also saved as the
session's `source.py`, SHA `a787836f…1620946`. Its lines 65/69 set A/B K to 256;
lines 308–317 and 346–356 request stage 2. S's `candidate.py`, SHA
`c6e217be…ab80e51`, changes only A/B K to 128 at lines 65/69 and keeps these
pipeline requests. S's `collection.json` matches that candidate SHA and original
reference SHA, reports `passed=true`, `swimlane=0`, `pmu=0` and the default
fixture flags. B's older manifest has `swimlane=4`, `pmu=0` and no explicit
`fixture` field; do not backfill that missing field as a newly recorded contract.

**Hints.** B's hint log contains four PH-MR-001 records in `Right`, including
the two `proj_a_mm` groups at lines 1–2 and `proj_b_mm` at 3–4. S's existing hint
log contains **zero** PH-MR-001 records. This is a checked log difference, not an
assertion that a missing log means no warnings. Candidate physical timing and
PMU were not collected; do not reuse B's row timings as S's measurements.

**The important correction:** both builds already allocate **two 32 KiB Right
buffers** in `proj_a_mm`. A packet saying “one allocation became two” is false.
Use the exact allocation and outer-loop fragments, not just pointer counts:

```python
# B/passes_dump/37_after_MemoryReuse.py:160,162
mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
mem_right_11: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
# B:168–170
for kb_inline36__idx_v0, (acc_a_inline152__iter_v1,) in pl.range(0, 16, 2, init_values=(acc_a_inline152__tile,)):
    k0_inline136__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline36__idx_v0 * 256
    k0_inline136__ssa_v0_1: pl.Scalar[pl.INDEX] = (kb_inline36__idx_v0 + 1) * 256
```

```python
# S/capture/passes_dump/37_after_MemoryReuse.py:159–160
mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
mem_right_11: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
# S:164–166
for kb_inline176__idx_v0, (acc_a_inline34__iter_v1,) in pl.range(0, 32, 2, init_values=(acc_a_inline34__tile,)):
    k0_inline33__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline176__idx_v0 * 128
    k0_inline33__ssa_v0_1: pl.Scalar[pl.INDEX] = (kb_inline176__idx_v0 + 1) * 128
```

The following is a **derived association**, with its raw fragment coordinates:

| Version | First outer chunk | Adjacent outer chunk | Meaning established |
| --- | --- | --- | --- |
| B, K256 | Mat `mem_mat_5`, then Right `mem_right_9` and `mem_right_11`: lines 174–175, 194–205 | Mat `mem_mat_7`, then the **same two** Right allocations: 180–181, 225–236 | Two 128-K L0 slices of each 256-K chunk reuse the Right pair across outer chunks |
| S, K128 | Mat `mem_mat_5`, then Right `mem_right_10`: 170–171, 185–192 | Mat `mem_mat_8`, then distinct Right `mem_right_11`: 179–180, 194–201 | The two adjacent outer 128-K chunks refer to different Right allocations |

This follows generated allocation/use edges. It does not prove actual pipeline
overlap, utilization or independent end-to-end benefit. Four warnings becoming
zero is not overlap proof either. The two builds' identical compiled-meta hash
makes a useful identity trap alongside these nonidentical IR/log hashes.

**Questions:** which artifacts apply to the selected source; how do the two
adjacent outer K chunks map to Right allocations; can an old serialization hint
be asserted for the candidate; what device overlap remains unmeasured? Do not
score merely counting two allocations as solving the relation question.

**Controls:** old B hint/IR supplied with candidate SHA must stay explicitly
historical, not become current candidate evidence. If S's log/IR is withheld or
its supplied digest mismatches, the candidate association is unavailable.
Never mutate files or relabel old bytes to create this control.

### I2. C4 M256 failure: distinguish a capacity warning from a proven cause

F's candidate source SHA is `5689a216…979e5ff`; M256/N256 implies a declared
FP32 accumulator of `256×256×4=262144 B`, **256 KiB**. The A2A3 Acc/L0C reference
limit is **128 KiB** in the canonical performance guide. F's preserved hint log
has **16 PH-AT-006 records**: oversized output/slicing is left untouched by the
current automatic tiling patterns. The last retained pass dump is
`37_after_MemoryReuse.py`. The attempt's `result.json` records return code 1,
no timeout and a generic candidate-process error. `device/stdout.log` contains
compile start/done followed by `ERR99999 UNKNOWN applicaiton exception`;
`device/stderr.log` is empty. No successful candidate validation manifest or
raw benchmark is available for this attempt.

**Questions:** which source produced these diagnostics; which capacity risk has
direct evidence; is numerical failure, an exact failing pass or a compiler bug
proven? Correct answer separates known warning/failure from **unresolved root
cause**. “Last retained pass” is not a fabricated precise exception location.

**Controls:** withholding the raw exception keeps the cause unknown; a real
older PASS manifest with another source SHA is not this candidate's PASS.
Withholding hints cannot be replaced by “no warnings.” Do not add a fake
precision failure or a synthetic compiler error to this real packet.

## 4. Work/resource priority: two packets

### W1. C4 M112/N256/20 lanes: 40 units, one submit, unmeasured candidate rows

C's source lines 49/55 establish HEAD_DIM=512 and OUT_DIM=1024. Lines 78–80 set
M112/N256/20 lanes. At the first 512-row chunk (`base=0`, `rows=512`), lines
163–179 give the exact work mapping:

```text
scratch_rows = rows + min(base,16) = 512
lane_blocks = ceil(512/112) = 5
lane_units = lane_blocks * (1024/256) * 2 = 40
lane_steps = ceil(40/20) = 2
unit = lane + si*20; wsel = unit%2; rest = unit//2
sblk = rest//4; nblk = rest%4
```

Every lane has two units for this chunk, not one. Generated orchestration
`C/capture/orchestration/_compressor_ratio4.cpp:118–128` sets block count 20
and performs **one `rt_submit_aic_task(1, params_t1)`** per outer chunk. The
40-block AIV pool submit at lines 130–152 is a separate task and must not be
counted as projection units or AIC cores. This is static launch evidence;
C's successful manifest explicitly says `swimlane=0`, `pmu=0`.

A's original level-4 capture provides an independently named baseline:
`deps.json` task **4294967297**, kernel IDs `[1,-1,-1]`, projection `block_num=32`.
The official reader finds **32 physical projection rows on 20 distinct AIC
cores**; 12 cores have two rows, eight have one. The metadata has 20 AIC and
40 AIV cores; name-map callable 1 is `prefill_c4_proj`. The projection's observed
envelope is 109.14→224.84 µs, or 115.70 µs. These are **baseline** observations,
not candidate assignment/timing. Existing pfdb TASK already exposes block_num,
num_rows and kernel_ids; rows/core expose the underlying timing records.

The same formula on the original M128/N256 source gives **32 units**, not 16.
Changing only lanes 32→20 keeps 32 units: 12 lanes have two units and eight one.
This is a useful source-parameter counterfactual, not a second measured run.
M112's row starts clamp from nominal `[0,112,224,336,448]` to
`[0,112,224,336,400]` at this first chunk (source 177–179), yielding 48 repeated
row positions across projections. This does not establish all dynamic-chunk
or concurrent-write correctness; preserved comparator coverage is narrower.

**Questions:** distinguish logical submit, block grid, loop unit, physical row
and core; recover 40 units/two per lane; identify which counts are observed
only on the baseline and which candidate measurements are unavailable.
The later stopping proposal in `…/c4_agent-0-semantic_contract/attempt-1/result.json`
incorrectly says 20 units and M128=16. It is reviewer evidence of the error,
not a model-visible authority or expected-answer snippet.

**R association fields:** source/workload IDs; scope=first 512-row chunk;
M/N/OUT_DIM/weights; formula and evaluated unit counts; lane unit distribution;
one generated AIC submit/block_num=20; baseline task/row/core IDs;
candidate observed rows/core map=`unavailable`. No measured MAC/HBM count is
deduced from source unit count. If dynamic values or source mapping are omitted,
answer symbolically or unknown rather than extrapolate the fixed 512-row result.

### W2. SWA Right versus Vec: do not merge resource pools

Use G's actual PH-MR-001 lines 1–2, naming **Right** and the requested/fitted
buffer count. Pair them with the original source `GS/source.py:220–233`:
its identity `pl.mul(acc_out,1.0)` is in the output path. The canonical type
guide lines 339–361 maps **Right→L0B**, **Vec→unified buffer**, **Acc→L0C** and
states that elementwise operations use Vec. G's generated
`kernels/aic/o_proj_a_quant_aic.cpp:497,624–635` contains Acc TPUSH, Vec TPOP and
TMULS, making the placement concrete rather than merely assuming a DSL op's
cost. This exposes separate operand staging, accumulator transfer and vector
work; it does not provide an allocation proof that shrinking Vec restores a
Right pipeline buffer.

**Questions:** identify the warning's exact space and affected operand path;
does removal of a vector temporary by itself prove relief of the warned pool;
does arithmetic identity certify the mixed-lane handoff? Correct answers are
Right/L0B, no demonstrated cross-pool capacity relief, and no synchronization
certificate. The existing failed-candidate review gives further mechanism
evidence, but do not add its causal conclusions as new “proven” input.

**Controls:** a packet containing only the Right hint and source cannot infer
the exact new buffer placement. Withholding the memory-space mapping requires
an explicit missing definition unless the same mapping was already provided
in both arms. A change in source or generated path invalidates the old mapping
as current candidate evidence; same operation spelling is insufficient.

For a pure relation-presentation comparison, put the same raw source, generated
instruction excerpts and type-guide mapping in D1 and R. R may link them by
operand/space IDs; it may not uniquely add the mapping and call that formatting.

## 5. Existing reuse and minimal seam if experiments support it

| Available today | What it does not yet establish | Smallest justified reuse |
| --- | --- | --- |
| `inventory --analysis-details`, artifact SHA and immutable-file resolution | Artifact identity is not a source/build/workload equivalence proof | Keep these provenance/status paths; prototype candidate evidence manifest externally first |
| TASK block_num/num_rows/kernel_ids, ROW/core, official runtime converter | Kernel-local loop units, unit→block relation, candidate row mapping without a trace | Reuse existing facts and exact source excerpts; only add a tiny anchored relation summary if the controlled test helps |
| `perf_hints` verbatim source/origin; `memory` kernel/space/usage/limit | Hint→current candidate build and outer-chunk→allocation-use relation | Preserve compiler text; a local source/build selector/index is narrower than a new top-level tool or IR engine |
| trial parent/hypothesis/evidence_refs and compile_error without a run | Typed source/build/validation binding, or compile-only evidence as a profiled run | Use explicit evidence references to the research manifest now; a validated artifact-group extension can be considered only after utility evidence |

Initial ingest discovers hints/memory alongside a **complete core capture**
(`ingest/__init__.py:179–203,228–231`). Generic `attach` discovers args/scope/
incore/host/dispatch artifacts, **not compiler hints, memory reports, source,
compiled metadata, exceptions or PMU-only data** (`ingest/analysis.py:27–75`).
These candidate runs used swimlane 0; a failed/compile-only candidate cannot
be made a normal pfdb run by inventing records/deps/name-map files. Do not
attach its new hints to the old run and silently replace the baseline build.

If the experiments demonstrate value, first keep an offline build-group
manifest plus small indexed excerpt payload, use existing query/lifecycle
navigation for provenance, and preserve a separate candidate identity when no
profile run exists. Decide the narrow public selector after that result;
this audit does not prescribe a universal association schema or new command.
Runtime remains authoritative for its timing and analysis; local source
arithmetic and explicit excerpt relations must be labeled as such.

## 6. Existing attach/trial safety limits

- `attach` checks the run exists and hashes each registered file, but trusts the
  caller's chosen run. It does not compare source/build/workload identity before
  replacing an optional artifact kind. Good checksums and transactions prevent
  byte corruption; they do not prove that another build's artifact belongs to
  this run (`ingest/analysis.py:77–116,226–290`).
- `refresh --source` checks the original core records/deps/name-map hashes and
  matching existing optional hashes. It explicitly permits replacement via
  attach; refresh is not an API for reinterpreting an old run as a new candidate
  (`ingest/analysis.py:292–326`).
- Trial binding now refuses a different run once bound and protects referenced
  IDs from immediate reuse. However, the **first** bind only verifies that the
  run exists. `changed_files` contains paths, evidence_refs/verdict are caller
  supplied, and neither `bind_trial` nor `set_verdict` validates source digest,
  workload, PASS coverage or a diagnostic attachment (`lifecycle/trial.py:30–85`;
  schema `0001_init.sql:213–224`). Use a new linked child trial for a new run;
  do not treat that lineage alone as proof of matching source or legality.
- Source SHA, build-file digests and collection manifests must stay independent
  checks. The real identical-compiled-meta example above defeats signature-only
  matching. Negative controls should produce stale/mismatched/unavailable
  states, not silently borrow a benchmark, PASS, hint or task ID from a neighbor.

## 7. Exact file anchors

Paths below use section 2 aliases. Original byte hashes include compiler paths;
public excerpts should use consistent redaction, without changing these anchors.

| File | SHA-256 |
| --- | --- |
| models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py (also session `source.py`) | `a787836fb363c4b560dcb60a625f94df3a6dd5280a298a915135ef3ac1620946` |
| B/report/perf_hints.log | `61c489f5ca3ad71cd23974f337f5c535e08a6cf25dd9af88447373a7911e033e` |
| B/passes_dump/37_after_MemoryReuse.py | `108a3ffa51a9b58390f5fba5bdad38d9482dc1569613b4ef6c36ac7a30725545` |
| B/compiled_meta.json and S/capture/compiled_meta.json | `a818d7044dfea0a1e694286ef88aa534e389d2b97e483fb339b56228a2534ac0` |
| B/collection.json | `dc99bef66a8b11c1e0ff8c071948ed77705df72bae166e0bb8fccb3be789e80a` |
| S/candidate.py | `c6e217bec89e36f7ca9147c7320b305eff5b1e9b4563bfa92f2d9d7ceab80e51` |
| S/capture/collection.json | `4fd6bf32aab4668bcfeeaf1541bd064622c075c1fbe092ae87319145dc92610a` |
| S/capture/report/perf_hints.log | `8850f3e89aab581bab9a7d9fd2edfa145298e9fb8a9b66a9f6cdd53a686410d2` |
| S/capture/passes_dump/37_after_MemoryReuse.py | `4e0e1e151ec48efb924f470c43e1ab4aee8059bff6d38ef859cee542f0ff3a8e` |
| C/candidate.py | `525d0eee57cf2886667b2c2092cb37b59d50e190c34f6252193b646c62b2cab2` |
| C/capture/compiled_meta.json | `6ae56cbc0827cef3f7cda585ca5dbdcfbf89e8eec0c13f16ce3d6dc1b1f5bb5f` |
| C/capture/result_s0_p0.json | `6cce889e4ba5268c09084ffaf05ce403751af106091e12f79b2fcda367f7dd91` |
| C/capture/orchestration/_compressor_ratio4.cpp | `3e8e877e0d4bf31e4c66cd2b4b77f2e0e0fd6c0d6d126a30e29fd99ce9eaa074` |
| build_output/pfdb_development/optimization-pilot-v1/c4_agent-0-semantic_contract/attempt-1/result.json (proposal error only) | `b11565dc26a4cdbdab5704b82e084b4d749592675c10343e440b518b808bd65b` |
| A/dfx_outputs/deps.json | `52503a04533d3e598785f3edbfa27a87357467abf657b5aacbed36a29b3ca0bc` |
| A/dfx_outputs/chip_swimlane_records.json | `37e9164df5d2695cfd5550e916877e0a6114e480bfd19b9c4bd5cd54d01cc74c` |
| A/result_s4_p0.json | `0e41addc050b7dd8cd8c7c5a6a7da3d735214a6797cc7984f2639c87d595cded` |
| benchmarks/implementation/deepseek-v4.1-flash/deepseek_v4_flash_dspark/prefill_compressor_ratio4_v1.py | `4c7151590d7406116601949bc0604aaac4ee519c078a5c905b89f709a08196ba` |
| F/candidate.py | `5689a216c61e303056f8e71b55d55ddd34357a5d6c6975f5f1b510b96979e5ff` |
| F/result.json | `69394f6e3dad42cffb53383f89dcb5007b6d6789b493d63cf2328d688e136c9a` |
| F/capture/report/perf_hints.log | `ac43a76b51bc14bbd81a182f5aea101f584b0b56029b3f645774f5c10867632c` |
| F/capture/passes_dump/37_after_MemoryReuse.py | `b42a541f3cfeeaa6380d34000bc053ff0f28e1e5e93ec194924f4b63bd2d3ab0` |
| F/device/stdout.log | `d9bc06db8bf3f83a8a19e2dd7b4f85b6e811a3e08cd9dde304d5a65b1bc541ab` |
| F/device/stderr.log (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| G/report/perf_hints.log | `26ee1f338b87ecbaf370169cb7c58dc1699aac8267f839e8261c63d942c4e2b9` |
| G/kernels/aic/o_proj_a_quant_aic.cpp | `abf6fc025f2a37d9454590262c5b6bbe6d5f6040e89e63d7b92be9c4d82352e2` |
| GS/source.py | `0ed4e4bac3be06bd8b0b08b66ca49594435e6553180ff5cba3da9546200ff89b` |
| GS/attempt-0/result.json | `87e1fa403099ede437a11ee68ecb1cbace32d7e34d6361f9ee09f990d0fbd9b4` |
| docs/pypto-language/03-types.md | `73f335168c89fab846a66fa9d696def9b7f203c50b56cf3c0d4e3c2405211b23` |

The existing external producer implementations are
`profile_db/research/pfdb_development/collect_swa.py` SHA
`0ef3c926694f2e54d42cccf4400274d7a65ccd5dbf53896ac2adab42c97d489b`
and `benchmarks/implementation/deepseek-v4.1-flash/research_assets/compressor_ratio4_gap.py`
SHA `1ea79ebedde28fbea3d318723b67e2c0d49179c1001036286ec2ca3acfde727e`.
They explain the existing manifests and comparator scope; this audit did not
execute them. Preserve the pinned PyPTO/runtime/toolchain provenance alongside
each future packet rather than treating today's package version as a build ID.
