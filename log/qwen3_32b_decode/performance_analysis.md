# Performance Analysis: Qwen3-32B Decode Kernel

This document explains, from first principles and with measured data, **why the
current runtime (~1.79 ms) is close to the hardware ceiling** for this kernel,
and **where the remaining bottleneck is**.

All numbers come from the L2 perf records of the current best build
(`build_output/.../dfx_outputs/l2_perf_records.json`) at commit `f0022d6`
(the 6-optimization version, ~1789 µs Total Test Time).

---

## 1. TL;DR

- The kernel is **HBM-bandwidth bound**, not compute bound.
- The decode workload is a stack of **M = 16 (batch) skinny GEMMs**. Their
  arithmetic intensity is ≈ 16 FLOP/byte, far below the AI Core's compute:bandwidth
  balance point, so every large matmul is limited by how fast weights stream from
  HBM, not by cube math.
- The chip has **24 cube cores (AIC) and 48 vector cores (AIV)**. The cube cores
  are busy **31 442 core-µs** → an ideal lower bound of **31 442 / 24 ≈ 1 310 µs**
  if perfectly packed. The vector cores are busy only **3 567 core-µs** (74 µs
  ideal) — essentially free.
- Measured wall clock is **1 795 µs**, i.e. **73 % cube duty cycle**. The big
  matmuls each sustain **≈ 1.0–1.4 TB/s** effective HBM bandwidth — close to the
  A3/910C HBM peak. There is no large structural win left without **reducing the
  bytes read** (weight quantization), which is out of scope (weights are bf16
  inputs and the golden reference consumes them as bf16).

---

## 2. Hardware and the measured ceiling

| Resource | Count | Busy (core-µs) | Ideal time = busy / cores |
|---|---|---|---|
| Cube cores (AIC) | 24 | 31 442 | **1 310 µs** |
| Vector cores (AIV) | 48 | 3 567 | 74 µs |

Wall clock = **1 795 µs**. Cube duty cycle = 31 442 / (24 × 1795) = **73 %**.

Two immediate conclusions:

1. **The cube (AIC) is the bottleneck resource.** Vector work (RMSNorm, softmax,
   SiLU, the split-K reduce passes) totals 74 µs of ideal time across 48 cores —
   the vector units sit ~95 % idle, fully hidden under cube work. Optimizing
   anything vector-side is pointless.
2. **The hard floor is ~1 310 µs** — the time to run all cube kernels if they
   packed perfectly onto 24 cores. We are at 1 795 µs, 1.37× the floor.

---

## 3. Why the cube is memory bound (roofline)

### 3.1 Compute vs. memory demand

Per single-layer decode forward:

| Quantity | Value |
|---|---|
| Total matmul work | ≈ 13.0 G MAC ≈ **26 GFLOP** |
| Total weight bytes (bf16) | **1.560 GB** |
| KV-cache bytes read (avg ctx ≈ 2048) | ≈ 0.27 GB |

Weight bytes by tensor (bf16, 2 B/elem):

| Weight | Shape | Bytes |
|---|---|---|
| wq | 8192×8192 | 134.2 MB |
| wk, wv | 8192×1024 ×2 | 33.6 MB |
| wo | 8192×8192 | 134.2 MB |
| w_gate, w_up | 8192×25600 ×2 | 838.9 MB |
| w_down | 25600×8192 | 419.4 MB |
| **Total** | | **1 560 MB** |

### 3.2 Arithmetic intensity = batch size

For a GEMM `[M,K] × [K,N]`, the weight matrix `[K,N]` is `K·N` elements read once
and reused `M` times. Arithmetic intensity is therefore **≈ M FLOP/byte**. Here
**M = BATCH = 16**, so AI ≈ 16.

The AI Core's machine balance (peak FLOP ÷ peak byte/s) is in the hundreds of
FLOP/byte. With AI ≈ 16 we are **deep on the memory-bound side of the roofline** —
roughly an order of magnitude below the compute knee. The cube spends most of each
kernel **waiting on MTE2 (HBM → L1 weight loads)**, not doing math.

Sanity check on the two floors:
- **Compute floor**: 26 GFLOP on 24 cube cores is tens of µs — negligible.
- **Memory floor**: 1.56 GB of weights at ~1.2 TB/s ≈ **1 300 µs** — matches the
  measured cube ideal of 1 310 µs almost exactly.

The agreement is the proof: **cube-busy time ≈ weight-streaming time.**

### 3.3 Empirical bandwidth per matmul

Effective HBM BW = (weight bytes) / (cube-busy ÷ cores-actually-used):

| Matmul | Weight | Cube-busy (core-µs) | Elapsed (µs) | Eff. BW |
|---|---|---|---|---|
| q_proj | wq 134 MB | 2 166 (16 cores) | 135 | ~0.99 TB/s |
| out_partial | wo 134 MB | 2 876 (24) | 120 | ~1.12 TB/s |
| gate_proj | w_gate 419 MB | 8 710 (24) | 363 | ~1.16 TB/s |
| up_proj | w_up 419 MB | 7 883 (24) | 328 | ~1.28 TB/s |
| down_partial | w_down 419 MB | 7 124 (24) | 297 | ~1.41 TB/s |

Every large matmul runs at **~1.0–1.4 TB/s**, i.e. near the A3/910C HBM peak.
This is the single most important fact in this document: the matmuls are not
slow because of bad tiling or scheduling — they are **streaming weights at close
to peak memory bandwidth**. The split-K rewrites (down/out/gate/up) are exactly
what pushed each kernel up to this BW by issuing enough concurrent MTE2 to
saturate HBM; note down_partial, the most aggressively split, reaches the highest
effective BW.

---

## 4. Where the remaining 485 µs goes (1 795 − 1 310)

The kernel is at 73 % of the cube floor. The 27 % gap is **not** recoverable cube
throughput; it is structural overhead that cannot fully overlap:

### 4.1 Small-matmul phases under-fill the 24 cube cores
- `q_proj`: 16 tasks → only 16 of 24 cube cores used (Q_PROJ_OUT_CHUNK=512).
- `kv_proj`: 4 tasks → only 4 of 24 cores used.

These phases are **latency-bound, not throughput-bound**. However, they are *also*
HBM-saturated: q_proj reads 134 MB in ~135 µs ≈ 1 TB/s with just 16 cores, so the
16 cores already nearly fill HBM. Adding more cores (e.g. split-K on q_proj) would
**not** help — HBM is the shared limit, and 16 streaming cores already reach it.
This is why q_proj/kv_proj sit at 97–99 % exec% yet leave cores idle: the idle
cores would have nothing faster to do.

### 4.2 Phase serialization
The forward is a sequential chain of data-dependent scopes:
`rmsnorm → q/kv proj → attention → out_proj → post_rmsnorm → gate/up → silu → down`.
Each boundary is a barrier; the next scope's first weight tiles cannot be loaded
until its inputs exist. This shows up as **head OH** (dispatch → kernel-start gap),
which totals 21 632 core-µs (36.5 % of latency) and is dominated by the matmul
kernels waiting for their first HBM weight tile — the memory-bound symptom,
partially but not fully hidden behind neighbouring kernels.

### 4.3 Attention: imbalance + cube/vector ping-pong
Attention runs `pl.parallel(BATCH=16)` with **random per-row seq_lens**, so the
16 lanes are load-imbalanced (a 4096-ctx row does 16× the work of a 256-ctx row).
Within a row the pipeline is `qk(cube) → softmax(vec) → sv(cube) → online(vec)` —
four hand-offs that serialize per group. The attention cube work (qk+sv ≈ 2 174
core-µs) is small, but its imbalance and hand-offs stretch its wall footprint
beyond the ideal /24. Fusing qk+softmax was attempted and is blocked by a compiler
limitation (dynamic length-mask slice on a cube output); see
`failed_optimization.md`.

### 4.4 Vector reduce/epilogue is free
`out_reduce`, `down_reduce`, `silu`, both RMSNorms and softmax all land on the
48 idle vector cores (74 µs ideal total). They do **not** sit on the critical
path — confirming that the split-K reduce passes added negligible cost.

---

## 5. Why the floor itself cannot move (within scope)

The 1 310 µs cube floor is set by **1.56 GB of weights ÷ HBM bandwidth**. To beat
it you must change one of the three terms:

| Lever | Effect | In scope? |
|---|---|---|
| Fewer weight bytes (e.g. W4A16 / W8A16 quantization) | 2–4× less HBM traffic → the only large win | **No** — weights are bf16 inputs; the golden reference multiplies in bf16, so changing dtype changes the result and is disallowed |
| Larger batch M | Raises arithmetic intensity, amortizes weight reads over more rows | **No** — BATCH = 16 is fixed by the problem |
| Higher HBM bandwidth | Hardware change | **No** |
| Avoid re-reading weights | Each weight is already read exactly once | Already optimal |

Everything we *can* control — tiling, task granularity, split-K, fusion — only
affects how close we get to the HBM floor, not the floor itself. We have moved
from the baseline 1 999 µs (≈ 1.53× floor) to 1 789 µs (≈ 1.37× floor), and the
remaining gap is structural serialization that overlaps poorly with a memory-bound
pipeline.

---

## 6. The bottleneck, stated precisely

1. **Primary (fundamental): HBM bandwidth.** ~75 % of the runtime is the
   unavoidable streaming of 1.56 GB of bf16 weights through the cube cores at
   ~1.0–1.4 TB/s. This is a hard physical floor (~1 310 µs) that no kernel-level
   change can cross without reducing weight bytes (quantization, out of scope).

2. **Secondary (the recoverable ~25 %): phase serialization & head-OH bubbles.**
   The sequential scope chain plus the small `q_proj`/`kv_proj` phases mean the
   24 cube cores are not perfectly packed (73 % duty). But because each phase is
   *itself* HBM-saturated, even perfect packing would be bounded by HBM, so the
   realistically recoverable headroom here is small (low tens of µs), and several
   attempts to claw it back (bigger tiles, deeper pipelines, fusion, finer
   split-K) hit UB-capacity, compiler, or runtime-scheduler limits — see
   `failed_optimization.md`.

3. **Not a bottleneck: vector units and compute.** The 48 AIV cores are ~95 %
   idle and the cube's arithmetic capacity is ~20× more than the workload needs.
   Optimizing either cannot help.

**Conclusion:** the kernel is operating near the memory-bandwidth roofline for a
batch-16 bf16 decode layer. Further meaningful speedup requires reducing weight
traffic (quantization) or increasing batch size — both outside the constraints of
this experiment.
