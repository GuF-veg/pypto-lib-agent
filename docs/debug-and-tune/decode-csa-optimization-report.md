# DeepSeek-V4 DSpark CSA Tuning Report

## Outcome

No algorithmic implementation was retained. The final kernel is numerically
identical to the campaign baseline; the only retained source change adds
opt-in capture controls to `decode_csa.py`. This is an intentional result:
none of the investigated data-flow transformations could be compiled by the
available A2/A3 high-level PyPTO/PTOAS path while preserving the contract.

The campaign used pfdb run `1` in `.pfdb/decode-csa-tuning.duckdb`. All three
registered hypotheses are closed as `regression`; none dispatched a correct
candidate, so no trial is presented as a performance result.

## Reproducibility

| Item | Value |
| --- | --- |
| Repository commit | `f540ca263f53d65534037653010e938d79e50dc8` |
| Python / environment | Python 3.11.15, conda environment `pypto` |
| PyPTO package | 0.1.0 |
| PTOAS | `/data1/home/gufeng/project/ptoas-v0.57` |
| Platform / device | `a2a3`, Ascend 910B4, device 0 |
| Workload | standalone CSA decode, B=16, S=8 |
| Frozen golden | `build_output/_jit_attention_csa_test_20260828_152331/data` |
| Profiled capture | `build_output/_jit_attention_csa_test_20260828_153126/dfx_outputs` |

The final replay used the frozen input and output cache. It passed both public
mutable results: `kv_cache` and `x_out`. The normalized AST hashes remained
unchanged throughout the retained implementation:

```text
golden_attention_csa: e5bfaaeb6f49747b19ee5414c9a3606a409ccf31aca93e5b5d65029b36609f70
build_tensor_specs:   617f24a306880dfd23285ddace8653378373293c72d841dac11f5c52f38e18ef
```

## Baseline Evidence

The authoritative unprofiled benchmark is 300 pooled rounds from three
100-round invocations with frozen B=16 data:

| Minimum | Median | Mean | Maximum |
| ---: | ---: | ---: | ---: |
| 2276.1 us | 2414.9 us | 2410.1 us | 2533.0 us |

pfdb run `1` is level 4 and contains 71 logical tasks, 3,482 physical rows,
229 dependency edges, and a 3749.94 us profile makespan. Its observed critical
chain contains these material stages:

```text
score -> topk -> csa_slots_build_valid_qk_plan -> qk_pv -> merge_norm -> proj_a_mm
```

| Task family | Busy time | Wall time | Why it mattered |
| --- | ---: | ---: | --- |
| `score` | 480.84 us | 506.24 us | Produces dense 4,096-element score rows. |
| `topk` | 98.06 us | 112.18 us | Sorts and selects sparse attention indices. |
| `qk_pv` | 589.70 us | 590.66 us | Largest critical sparse-attention compute stage. |
| `merge_norm` | 376.76 us | 386.08 us | Merges the QK/PV online-softmax partial state. |
| `proj_a_mm` | 543.96 us | 550.26 us | Required downstream grouped projection. |

Raw source and profile inspection show that `qk_pv` materializes FP32
`mi [40960, 1]`, `li [40960, 1]`, and `oi [40960, 512]` before `merge_norm`.
The output projection also requires a complete per-group output range before
INT8 quantization. These are data-flow boundaries, not tile-size issues.

## Trials

### Trial 1: Persistent sparse-attention online softmax

**Hypothesis.** Keep `mi`, `li`, and `oi` in persistent on-chip state over the
sparse K blocks, removing the `qk_pv -> merge_norm` GM handoff.

**Evidence.** `qk_pv` and `merge_norm` are adjacent critical-path tasks with
590.66 us and 386.08 us wall time. The scratch tensors above are therefore a
large, justified target.

**Result.** Rejected before device dispatch. The high-level `pl.matmul` result
cannot be kept as vector state across the next sparse block. The required
`row_expand_mul` consumes Vec state, while the result is Acc state;
`pl.move` is invalid for this high-level TensorType. The same A2/A3 limitation
is documented by the existing SWA/HCA implementations.

**Verdict.** `trial_id=1`, `regression`. A correct implementation requires a
lower-level persistent kernel or an explicit Acc-to-Vec capability, neither of
which is available in the allowed source path.

### Trial 2: Shared compact indexer helper

**Hypothesis.** Let the two existing score lanes locally sort 2,048 values,
persist only exact top-512 `(score, index)` pairs, then merge those two lists.
This removes the dense `[T, 4096]` FP32 score handoff for CSA while retaining
the standalone indexer score output.

**Result.** Rejected before device dispatch. `@pl.jit.inline` does not support
a Python bool parameter; internal temporary tensors passed to another inline
helper do not retain inferred metadata; and a closure static flag is undefined
to the DSL parser.

**Verdict.** `trial_id=2`, `regression`. The helper factoring itself was not a
viable representation of the algorithm.

### Trial 3: Dedicated static CSA indexer

**Hypothesis.** Duplicate the static CSA-only indexer body to avoid Trial 2's
helper-boundary limitations. Score lanes would own contiguous 2,048-position
ranges, use native `UINT32` sort indices beginning at 0 and 2048, and emit
only top-512 pairs to a final two-way merge.

**Result.** Rejected before device dispatch. The direct implementation exposed
several A2/A3 code-generation limitations:

- `UINT32 -> FP32` and `INT32 -> UINT32` tile casts have no native path.
- Unsigned scalar arithmetic used to construct interleaved indices is rejected
  by PTOAS.
- A dynamic 32-wide score tile cannot be assembled into a mutable 2,048-wide
  local tensor: `pto.tmov` requires matching non-mat source and destination
  shapes on A2/A3.
- `pl.assemble` produces the same shape failure as subscript assignment.

The trial was fully reverted. The source never dispatched, so it has no
correctness, benchmark, capture, or confidence interval result.

**Verdict.** `trial_id=3`, `regression`.

## Retained Change

`models/deepseek_v4_flash_dspark/decode_csa.py` now exposes opt-in
`--save-data`, `--enable-dep-gen`, `--enable-pmu`, `--enable-dump-args`, and
`--enable-scope-stats`, forwarding them to `run_jit`. Defaults are unchanged.
This made the baseline capture reproducible without changing the CSA math or
public tensor contract.

## Stop Condition

The evidence-backed targets above are the only candidates with a removable
critical contribution materially above observed benchmark noise. Each requires
state to survive a high-level compute boundary:

- score compaction requires a mutable wide local selection state;
- persistent sparse attention requires Acc-to-Vec state retention; and
- grouped projection requires full group-range information before mandated
  quantization.

The first two were attempted and blocked by compiler semantics. The projection
boundary is inherent to the existing quantization contract, and changing it
would change numerical behavior. Therefore no allowed, correct algorithmic
candidate remains in this source boundary. A future campaign should begin only
after adding a supported persistent mixed cube/vector primitive or authoring a
validated lower-level kernel.
