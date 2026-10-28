# Read-only review of two failed online candidates

The first candidate is **numerically invalid on the tested frozen SWA fixture**. Existing artifacts show a changed cross-core dataflow, but do not establish whether responsibility lies in an unsupported source pattern, dependency opt-out obligations, or a compiler defect. The second candidate has an opaque compile-stage failure. Neither is a performance result or evidence that runtime timing definitions are wrong.

No source, upstream checkout, capture or database was modified for this review. No candidate was recompiled or rerun. SHA-256 anchors for both candidate sources/diffs, generated output, retained diagnostics and the compiler source are saved in `build_output/pfdb_development/online-failure-review.json`.

## Candidate 0: removing `pl.mul(acc_out, 1.0)`

Evidence directory: `build_output/pfdb_development/optimization-pilot-v1/swa_agent-0-semantic_contract/attempt-0/`.

The complete source diff changes one GM scratch assignment from `pl.mul(acc_out, 1.0)` to `acc_out`. The original source SHA begins `0ed4e4bac3be06bd`; candidate SHA begins `b53369fa833e1f95`. The original passed, while this candidate's preserved `capture/collection.json` reports **31,555/32,768 mismatches (96.2982%)**. The first ten reported mismatches have actual value zero; this excerpt does not prove that every output was zero. Both use the existing frozen golden directory; the candidate contract remains T=8, H=64, head dimension 512, window 128. The comparator is unchanged (`atol=1e-4`, `rtol=1/128`, default allowed error ratio 0.005). This is far beyond a tolerance-boundary failure.

The relevant lowered programs differ materially:

| Original generated function | Candidate generated function |
| --- | --- |
| AIC produces the accumulator and `TPUSH`es it | AIC writes the accumulator to GM with `TSTORE` inside the four-iteration N loop |
| AIV performs `TPOP`, the identity `TMULS`, then the GM scratch store | AIV loads the GM scratch independently before amax/quantization |
| The transfer supplies a cross-core ordering operation | No cross-core handshake for this scratch edge appears in the split IR or generated projection functions |

Original evidence: `build_output/pfdb_development/swa_agent_capture/kernels/aic/o_proj_a_quant_aic.cpp:497` (`TPUSH`) and `:624` (`TPOP`). Candidate evidence: `capture/passes_dump/25_after_ExpandMixedKernel.py:19` (store loop), `:46` (Acc→GM store), `:59` (AIV load), and `capture/kernels/aic/o_proj_a_quant_aic.cpp:544` (`TSTORE`). The candidate orchestration still submits these lanes together as `MixedKernels {4,5,5}`. Source order in the high-level region does not by itself demonstrate a hardware happens-before edge after splitting.

This is a concrete mechanism to investigate, **not proof that a race alone caused the observed numbers**. No intermediate scratch values, hardware access ordering or isolated reduced control was collected.

### Three gates and the ownership boundary

1. **Style/primitive contract:** direct accumulator writeback is supported in principle. The [memory-space rules](../../../docs/pypto-language/03-types.md#memory-spaces) allow `Vec` and `Acc` store sources; [operations](../../../docs/pypto-coding/operations.md#plslice--plassemble--loadstore-at-the-plat-boundary) documents slice assignment as the assemble/store form. The unchanged K loop retains `init_cond=(kb == 0)` and the explicit pipeline carry required by the [matmul guide](../../../docs/pypto-language/17-matmul.md). This does **not** establish the complete mixed GM handoff's synchronization contract.
2. **Pins:** read-only checks matched PyPTO `ee49fceaa3290cf90257a22eab867c04245b499d`, runtime pin/checkout `6e383fc57c12a8bd4dce045289b8b4f5d55a15f1`, PTOAS installed/pinned 0.65, and PTO-ISA pin/checkout `c0d7148e95ef73bd12a73165fdce4b723a3b7e72`. The configured CANN root names 9.0.0; this review did not independently recover its version metadata.
3. **Own-code/contract:** tensor shapes, dtypes, golden and comparator are unchanged; no removed rounding conversion explains the zeros. However, the scratch retains `manual_dep=True`. The [dependency guide](../../../docs/debug-and-tune/dependency-and-scheduling.md#turning-inference-off) treats opt-outs as caller assertions, and [tensor operations](../../../docs/pypto-language/07-tensor-and-system-operations.md#plcreate_tensor) says the tensor leaves automatic dependency tracking for its lifetime. The guide does not explicitly settle this internal mixed-lane GM edge. Its safety must be established before calling the entire transformation sanctioned.

Primary compiler source narrows the uncertainty. At the pinned revision, `src/ir/transforms/expand_mixed_kernel_pass.cpp:1282–1292` explicitly recognizes GM-mediated AIC-store/AIV-load dependencies and inserts a synchronization handshake for supported shapes. `CollectGmCrossLaneSyncs` is deliberately conservative: the producer store's structural body must be the same as, or an ancestor of, the consumer load's body (`:1324–1332`, `:1490–1499`). It also resolves store-result aliases rather than all loop-carried aliases (`:1305–1318`, `:1337–1359`). Here the store is **inside** the N loop and the initial consumer load is **outside** that loop, so the observed missing handshake is consistent with a case outside that matching rule. The matcher itself does not consult `manual_dep`; that fact alone does not define the public opt-out contract.

The review therefore leaves **fault ownership unresolved**. The primitive is accepted; this complete transformation is not proven safe on the current toolchain. A compiler miscompile verdict would require settling the manual-dependency contract and isolating the unsupported control-flow/alias shape. The unresolved own-code gate is not bypassed by mathematical identity, and no upstream defect is asserted here.

The smallest useful follow-up evidence would be a reduced mixed region with an initialized producer, a loop of GM stores, and an outside-loop consumer; explicit comparison of automatic vs opted-out dependencies; source/generated synchronization proof; and intermediate outputs showing which read becomes stale. A compile-only reduced control can locate missing lowering, while an ordering/precision claim still needs a separate device validation through the external harness. This review created neither a workaround nor a new tuning candidate.

### Agent reasoning

The proposal correctly cited PH-MR-001, but made two unsupported jumps:

- The hint reports pressure in **Right/L0B**, while the removed identity multiply is a **Vec/UB** operation. These are separate memory spaces in the [type guide](../../../docs/pypto-language/03-types.md#memory-spaces). No supplied allocation evidence shows that removing this UB temporary restores a second Right buffer. Compiler scheduling might change indirectly; that requires evidence rather than a shared-capacity assumption.
- Exact arithmetic equivalence was treated as a complete implementation invariant. Removing a vector consumer also removed the original Acc→Vec transfer path and exposed a GM-mediated cross-lane handoff. Numerical correctness was not established by the proposal's invariants.

This is useful negative evidence for pfdb design: runtime duration semantics can help identify a region, but cannot certify a legal compiler transformation. Resource hints need explicit memory-space identity, and generated placement/transfer evidence must remain reachable when an apparently trivial algebraic edit changes execution structure.

## Candidate 1: splitting projection chunks and quantization

Evidence directory: `build_output/pfdb_development/optimization-pilot-v1/swa_agent-1-semantic_contract/attempt-0/`.

The proposal creates 32 projection tasks from eight groups × four N chunks, separates quantization, and removes the scratch's `manual_dep`. Its public hypothesis calls these “32 blocks”; the source actually introduces **32 logical task submissions**, whose blocks and physical rows require generated evidence. It also assumes automatic tracking provides the intended ordering without serializing the writes; that is not verified by the source change.

Retained pass dumps stop at `15_after_FlattenTileNdTo2D.py`; there is no generated executable or validation manifest for this candidate. `device/stdout.log` contains compile start/done and an opaque `ERR99999 UNKNOWN application exception`, with no useful exception text in `stderr.log`. The harness's [stage context manager](../../../golden/runner.py) (`_Stage`, line 138) prints “done” from `__exit__` even when an exception propagates, so that line does **not** establish successful compilation. No device numerical result can be assigned.

Classify it as **failed before a usable capture, compile-stage cause unresolved**. Symbolic NZ slicing/layout and a compiler rejection are possible investigation directions, not diagnosed causes. The missing evidence is the actual compiler exception/pass verifier diagnostic, not more utilization counters. This candidate consumes its original experiment slot and carries no speedup claim.
