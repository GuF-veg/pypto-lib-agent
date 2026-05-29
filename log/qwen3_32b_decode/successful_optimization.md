# Successful Optimizations (qwen3_32b_decode)

Baseline: 1998.90 us (commit 08d3539, unmodified, 100 repeat / 5 warmup).

## 1. Enlarge MLP output chunk (MLP_OUT_CHUNK 256 -> 512)

- **Result**: ~1998.9 -> ~1953 us (avg of 1941.95 / 1965.10), ~2.3% faster, PASS.
- **Rationale**: gate_proj/up_proj each spawned 100 tasks (MLP_OUT_BLOCKS = INTERMEDIATE/256 = 100)
  with ~75-77 us head OH each — together ~86% of all head overhead in the run.
  Doubling the output chunk halves the task count (100 -> 50 per projection), so the
  AICPU dispatches fewer, larger kernels and per-task head OH amortizes away. The cube
  handles [16,128]@[128,512] fine and 512 bf16 = 1024 B stays MTE-aligned.
- **Why it's a clean win**: single-constant change, no added complexity.

## 2. Enlarge down_proj K chunk (DOWN_K_CHUNK 128 -> 256)

- **Result**: ~1953 -> ~1920 us (avg of 1926.10 / 1915.11), ~1.7% faster, PASS.
- **Rationale**: down_proj is the largest exec consumer (~38% of total AICore time) with a
  200-step K reduction over INTERMEDIATE=25600 at DOWN_K_CHUNK=128. Doubling the K chunk
  halves the pipeline iterations (200 -> 100), reducing per-iteration loop/sync overhead
  while keeping the mlp slice [16,256] bf16 = 512 B aligned.
- **Why it's a clean win**: single-constant change.

## 3. Enlarge q_proj output chunk (Q_PROJ_OUT_CHUNK 256 -> 512, decoupled)

- **Result**: ~1920 -> ~1888 us (avg of 1881.98 / 1895.33), ~1.7% faster, PASS.
- **Rationale**: q_proj spawned 32 tasks (HIDDEN/256). kv_proj that follows showed ~94 us
  head OH from queueing behind those 32 cube tasks. Decoupling q_proj's output chunk from
  the shared Q_OUT_CHUNK (kept at 256 for out_proj) and raising it to 512 halves q_proj
  tasks (32 -> 16), cutting cube queue pressure so kv_proj dispatches sooner. wq slice
  [128,512] fills the cube and 512 bf16 = 1024 B stays aligned.
- **Note**: applied only to q_proj; out_proj still uses Q_OUT_CHUNK=256 (it is auto_chunk
  windowed and sensitive to chunk changes, see failed_optimization).

## 4. Split-K for down_proj (the biggest win)

- **Result**: ~1888 -> ~1820 us (avg of 1822.25 / 1818.57), ~3.6% faster, PASS.
- **Rationale**: down_proj reduces over K=INTERMEDIATE=25600 (K >> N=256 per block) — the
  classic K-dominant decode shape that under-uses cores. The old windowed auto_chunk
  version had only 16 windows. Rewriting as explicit split-K (DOWN_K_SPLIT=2) produces
  DOWN_PROJ_BLOCKS*2 = 64 partial-matmul tasks (each reducing half of K) plus a cheap
  32-task vector reduce (sum the 2 partials + residual). The 4x parallelism fills
  otherwise-idle cores during the most expensive phase. Online research (W4A16 / AscQLUT /
  HGEMM papers) all point to split-K as the key technique for K>>N decode GEMMs on Ascend.
- **Cost**: one extra GM tensor down_partial [2*16, 8192] fp32 (~1 MB) round-trip, negligible
  vs the weight traffic. Dropped the auto_chunk windowing for a plain 2-stage spmd, which is
  also conceptually clearer.
- **Note**: DOWN_K_SPLIT=4 fails codegen ("tmov address-space pair"); 2 is the working value.

## 5. Split-K for out_proj (same pattern as down_proj)

- **Result**: ~1820 -> ~1799 us (avg of 1805.67 / 1793.17), ~1.1% faster, PASS.
- **Rationale**: out_proj is also K-dominant (K=HIDDEN=8192 >> N=256 per block). Same split-K
  rewrite (OUT_K_SPLIT=2): 32*2 = 64 partial-matmul tasks + 32-task reduce that also folds in
  the hidden_states residual, producing resid1_tile. Smaller win than down_proj because K is
  3x shorter (8192 vs 25600) so there is less idle-core slack to recover, but still positive
  and the code mirrors down_proj.

## 6. Split-K for MLP gate/up projection + simplification

- **Result**: ~1799 -> ~1789 us (avg of 5 runs: 1782/1784/1801/1791/1786), ~0.6% faster, PASS.
- **Rationale**: gate/up are also K-dominant per output block (K=HIDDEN=8192 reduction,
  N=MLP_OUT_CHUNK=512). They showed only ~56% exec% (44% head OH = waiting on weight loads),
  meaning cores idled between loads. Replacing the nested parallel(25)x spmd(2) + per-group
  gate_group/up_group caching with a clean split-K (MLP_K_SPLIT=2): gate/up each become
  MLP_OUT_BLOCKS*2 = 100 partial-matmul tasks, then a 50-task reduce-and-SiLU pass. The extra
  concurrency overlaps more MTE2 weight loads, improving aggregate HBM utilization.
- **Bonus (simplification)**: removed MLP_SPMD_INNER, MLP_GROUP_CHUNK and the manual 2-block
  K prologue; the MLP now uses the same split-K shape as down_proj/out_proj. Small speed win
  AND simpler code.

## Summary

Baseline 1998.9 us -> ~1789 us, ~10.5% faster. The dominant lever was split-K (down_proj,
out_proj, gate/up) for the K-dominant decode GEMMs, exactly as the Ascend decode-GEMM
literature recommends. The remaining time is essentially HBM-bandwidth bound on the ~1.5 GB
of weights, so further large wins would require fewer weight bytes (e.g. quantization), which
is out of scope here.
