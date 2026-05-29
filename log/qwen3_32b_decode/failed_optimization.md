# Failed Optimizations (qwen3_32b_decode)

## OUT_PROJ_K_CHUNK 128 -> 256

- **Result**: ~1920 -> ~1943 us (regression), higher std (32) and max (2223). Discarded.
- **Lesson**: The down_proj K-chunk doubling helped, but the same trick on out_proj hurt.
  out_proj is windowed via auto_chunk + UP_DOWN split; doubling the K chunk likely
  perturbs the auto_chunk's internal pipelining (it already balances cube/vec windows),
  whereas down_proj benefits because its K reduction is much longer (200 vs 64 steps).
  Don't blindly apply K-chunk enlargement to every matmul — it depends on K-loop length
  and whether the region is auto_chunk-managed.

## MLP_SPMD_INNER 2 -> 5

- **Result**: ~1920 -> ~1933 us (mild regression). Discarded.
- **Lesson**: Bigger spmd fan-out for the MLP did not reduce wall clock; the larger
  per-group gate/up buffers and coarser windowing offset the dispatch savings. The
  MLP_OUT_CHUNK=512 / MLP_SPMD_INNER=2 balance is already near the sweet spot.

## Fuse MLP gate + up + SiLU into one SPMD body

- **Result**: at MLP_OUT_CHUNK=512 the fused loop FAILED memory allocation
  (AllocateMemoryAddr verification) — two live fp32 accumulators [16,512] plus the
  two interleaved K-loop ping-pong buffers exceed UB. Falling back to MLP_OUT_CHUNK=256
  compiled and passed but ran ~1979 us, worse than the separate-loop 512 layout (~1920).
- **Lesson**: Fusion removes intermediate gate/up buffers and task count, but it forces a
  smaller output chunk (256) due to UB pressure, and that loses more than the fusion
  saves. The separate gate/up/silu loops at MLP_OUT_CHUNK=512 win because the larger
  output chunk dominates. Keep gate/up/silu split.

## Tile-enlargement OOM cluster (UB is the binding constraint)

The following all failed `AllocateMemoryAddr` verification (UB out of memory), not a
logic error:
- DOWN_N_CHUNK 256 -> 512 (windowed down_proj: 2-stride => 1024 cols/window)
- DOWN_K_CHUNK 256 -> 512
- Q_PROJ_OUT_CHUNK 512 -> 1024
- KV_OUT_CHUNK 256 -> 512 (kv_proj holds two accumulators k_acc + v_acc)
- gate/up K-loop pl.pipeline stage 2 -> 4

**Lesson**: after MLP_OUT_CHUNK=512 + DOWN_K_CHUNK=256 + Q_PROJ_OUT_CHUNK=512, the UB is
saturated. Any further tile/stage enlargement OOMs. Future wins must NOT increase per-core
buffer footprint — look at reducing task count, redundant work, or improving overlap
without bigger tiles.

## Fuse SiLU into up_proj loop (remove up_group + silu loop)

- **Result**: OOM (AllocateMemoryAddr). The up_silu kernel must hold up_acc [16,512] fp32
  + K-loop ping-pong + read gate_acc [16,512] + sigmoid/mul temporaries — exceeds UB.
- **Lesson**: SiLU is kept as a separate spmd loop precisely because the up_proj kernel is
  already at the UB ceiling. Vector epilogues cannot be fused into the 512-chunk matmul
  kernels here. (The down/out_proj DO fuse cube+vector via auto_chunk windowing, which
  manages the split differently.)

## Fuse qk_matmul + softmax (remove all_raw_scores round-trip)

- **Result**: compile error — "tpop_from_aic dynamic valid_shape requires both valid_row
  and valid_col". Slicing the cube (aic) output directly with a dynamic valid_len for the
  causal/length mask is unsupported in a fused aic+aiv kernel.
- **Lesson**: all_raw_scores is materialized to GM specifically so the softmax can take a
  dynamic `valid_shape` slice (length masking). The cube->vector hand-off via GM is a
  compiler requirement here, not just a style choice. Attention is also only ~8% of exec
  and the kernel is ~95% HBM-bandwidth bound overall, so this round-trip is not worth
  fighting the compiler over.

## Over-splitting K (DOWN_K_SPLIT=4/5, OUT_K_SPLIT=4)

- **Results**: DOWN_K_SPLIT=4 -> codegen error (tmov address-space); DOWN_K_SPLIT=5 -> 1836 us
  (worse than 1799); OUT_K_SPLIT=4 -> 1841 us (worse than 1799).
- **Lesson**: split-K=2 is the sweet spot. Beyond 2, the extra partial tasks oversubscribe
  the cores (no more idle slack to fill) while the vector reduce pass and the larger
  down_partial/out_partial GM tensors add overhead. The win is from 2x parallelism, not
  unbounded splitting.

## MLP_OUT_CHUNK=256 combined with gate/up split-K

- **Result**: runtime TIMEOUT (300s task-queue kill). split-K already doubles task count;
  halving the output chunk to 256 yields 200 gate + 200 up + 100 silu tasks, overwhelming
  the AICPU scheduler.
- **Lesson**: with split-K active, keep MLP_OUT_CHUNK=512. Task-count explosion from
  combining small chunks with split-K is counterproductive (and can hang the runtime).
