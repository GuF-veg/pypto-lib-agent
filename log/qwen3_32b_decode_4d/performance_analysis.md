# Performance Analysis: Qwen3-32B Decode 4D Kernel (Updated Session 2/3)

Baseline 2072.91 us -> current best ~1944 us (commit 95f75a8), ~6.2% faster.
All changes PASS correctness validation.

## What changed (wins, in order)

1. **DOWN_K_CHUNK 64→128**: down_proj K-loop 400→200 iters. exec 388→369us.
2. **Q_PROJ_OUT_CHUNK 512** (decoupled): q_proj 32→16 tasks, exec% 63→97%, head OH 40→0.6us.
3. **KV_OUT_CHUNK 256→512**: k/v_proj 4→2 tasks each, ~97→98% exec%.
4. **K_CHUNK 128→256**: gate/up+post_rmsnorm K-loop 64→32 iters. The decisive effect was
   shrinking the post_rmsnorm tail-OH bubble (63→12us) on the critical path before gate/up.
5. **out_proj single 512-wide matmul** (drop lo/hi 256 split): exec 152→136us, simpler code.
6. **POST_RMS_K_CHUNK 512** (decouple from K_CHUNK): post_rmsnorm 32→16 iters, simpler code.
7. **out_proj K pipeline stage 2→3**: exec% 96.3→97.2%, marginal.
8. **k/v_proj K pipeline stage 2→3**: exec% ~97→98%.

## Current profile (commit 95f75a8, ~1944-1950us)

| Region               | Tasks | Exec/task | Exec%  | Head OH  | Notes |
|----------------------|-------|-----------|--------|----------|-------|
| gate_proj            | 100   | 75 us     | 54%    | 62 us    | HBM bw queue — bandwidth wall |
| up_proj              | 100   | 76 us     | 54%    | 62 us    | HBM bw queue — bandwidth wall |
| down_proj            | 16    | 368 us    | 99%    | 0.5 us   | M=16 latency bound, sweet spot |
| out_proj             | 16    | 139 us    | 98%    | 0.5 us   | optimal |
| q_proj               | 16    | 133 us    | 97%    | 0.5 us   | optimal |
| k/v_proj             | 2+2   | 125 us    | 97%    | 0.9 us   | optimal |
| attention (all)      | ~4400 | tiny      | 71-97% | low      | overlapped under cube matmuls |
| post_rmsnorm         | 1     | 18 us     | 63%    | 0.6 us   | 16-iter vector, tail OH ~10us |
| out_proj_residual    | 16    | 4 us      | 37%    | 0.5 us   | vector, tail OH ~7us |

## Hard ceilings hit

All paths to further improvement are blocked by hardware buffer limits:

| What we tried | Why it fails |
|---------------|-------------|
| DOWN_K_CHUNK=256 | Mat buffer OOM: [16,256]×[256,512]=540KB > 512KB limit |
| K_CHUNK=512 (gate/up) | Mat buffer OOM: [16,512]×[512,256]=557KB |
| OUT_PROJ_K_CHUNK=256 | Mat buffer OOM: [16,256]×[256,512]=540KB |
| MLP_OUT_CHUNK=512 | Mat buffer OOM: 540KB (B=[256,512]=256KB×2ping-pong+overhead) |
| gate+up+silu fused | Mat buffer OOM: two interleaved B-tiles=557KB |
| out_proj+residual fused | Vec buffer OOM (339-344KB > 192KB) even with UP_DOWN split |
| post_rmsnorm stage=4 | Vec buffer OOM: 237KB > 192KB |
| down_proj stage=3 | Worse (99% exec-bound; deeper pipeline overhead > benefit) |
| pl.split(UP_DOWN) for out_proj spmd | SplitVectorKernel: 4D acc dim-0=1 ≠ even |

## The fundamental constraint

- **HBM bandwidth floor**: gate+up alone stream 838 MB at ~1.34 TB/s ≈ A3/910C HBM peak.
  Every bf16 weight must be read; until weights are quantized (W4A16/INT4), the memory traffic
  cannot be reduced.
- **M=16 cube under-utilization**: batch=16 decode gives arithmetic intensity ≈ 16 FLOP/byte.
  The cube's 16×16 fractal means M=16 is the minimum, but still only 1/8 utilization of the
  full 16×16 tile. This is fundamental to small-batch decode.
- **512KB Mat / 192KB Vec buffer limits**: block all further tile enlargement or fusion attempts.

The kernel is at the bf16 bandwidth ceiling. The only path to substantially lower runtime
is weight quantization (W4A16/INT4), which is out of scope here (golden reference uses bf16).
