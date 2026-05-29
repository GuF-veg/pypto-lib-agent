# Failed Optimizations (qwen3_32b_decode_4d)

## MLP_OUT_CHUNK 256 -> 512 (tried before q_proj fix)
- Result: 2079 vs 2072 baseline, flat. Total exec dropped 1100us but wall unchanged.
- Lesson: gate/up region is HBM-bandwidth bound; per-task chunk size does not move wall
  time there. (Re-tested later on top of q_proj fix — see results tsv.)

## down_proj split-K=2
- Result: 2065 vs 2051, regressed. down_proj exec% fell 98.9 -> 77.5%, head OH 0.5 -> 55us.
- Lesson: unlike non-4d (whose down was a low-utilization windowed region), 4d's 16-task
  down_proj is already near its limit. Splitting K to 32 tasks just makes the 32 partials
  contend for the same w_down bandwidth and adds a reduce pass — net loss. The decode M=16
  matmul cannot be sped by more cube parallelism; it is latency/structure bound, not core-count bound.

## gate/up pipeline stage 2 -> 3
- Result: 2054 vs 2051, flat. exec% unchanged at 53%, head OH unchanged at ~64us.
- Lesson: the gate/up head OH is genuine HBM-bandwidth queueing, not pipeline-fill latency.
  Deeper software pipelining does not help a bandwidth-saturated matmul.

## down_proj DOWN_K_CHUNK 128 -> 256 (with or without N-split)
- Result: Mat buffer OOM. Plain: 540KB > 512KB limit. N-split into two 256-halves: 557KB
  (worse — two accumulator streams each ping-pong a weight tile).
- Lesson: the cube L0 Mat buffer is driven by DOWN_K_CHUNK (the weight tile's K dim).
  128 is the ceiling for down_proj; N-splitting does not relieve it because it doubles the
  number of live weight buffers.

## down_proj DOWN_N_CHUNK 512 -> 256 (more, smaller N-blocks to fill idle cores)
- Result: 2159 vs 2021, big regression. down_proj total exec 5915 -> 9056 core-us, exec% 98.9 -> 77.6%.
- Lesson: down_proj has only 16 tasks on 24 cores (8 idle), but "filling" the cores with
  smaller [16,128]@[128,256] matmuls makes each cube op even more underutilized (M=16 is
  already tiny). The cube strongly prefers fewer, LARGER N-tiles. DOWN_N_CHUNK=512 +
  DOWN_K_CHUNK=128 is the sweet spot; both split-K and smaller-N attempts to raise core
  occupancy backfire because they trade cube efficiency for parallelism the matmul can't use.

## OUT_PROJ_K_CHUNK 128 -> 256
- Result: Mat buffer OOM 540KB (B=[256,512]=262144B, ping-pong=512KB=exactly limit). 128 is ceiling.

## Post-rmsnorm pipeline stage 2->3 / 2->4
- stage=3: 1960 vs 1950us, flat (high std=38, noisy run). post_rmsnorm exec 18.8->16.4us but noise masks any win.
- stage=4: Vec buffer OOM 237KB > 192KB limit.

## rmsnorm pipeline stage 4->6
- Result: 1945/1953 across 2 runs, essentially flat. rmsnorm exec INCREASED 31->34us (more buffer management).
  No benefit, stage=4 is already optimal for the 64-iteration loop.

## q_proj K pipeline stage 2->3
- Result: 1960 vs 1950us (flat, noise). q_proj is 97.7% exec-bound, stage depth irrelevant.

## down_proj+residual fusion in one pl.at
- Result: AIV Vec buffer OOM 344KB > 192KB. Auto_chunk windowed AIV holds full down_acc
  [16,512]+resid+output simultaneously; 3 windows × 32KB = 96KB + staging = 344KB > 192KB.

## out_proj+residual fusion (3 attempts)
1. Slice 4D o_acc → FlattenTileNdTo2D error: slice not supported on >2D tiles.
2. Reshape o_acc to 2D + column slice → ptoas alignment error: strided column slice produces
   row_byte_size=4B (non-contiguous memory not representable as contiguous tile).
3. Assemble hidden_full + add whole o_acc → AIV Vec buffer OOM 340KB (windowed auto_chunk
   needs 3x buffers for 512-wide AIV operations: 3×32KB×3=288KB > 192KB).
4. spmd+UP_DOWN split → SplitVectorKernel error: requires even split dim, but 4D acc
   [1,1,16,512] has dim-0=1 (can't split into two halves). The 14b model uses 2D [BATCH,N]
   tensors that map cleanly to UP_DOWN; our 4D layout does NOT.

## Lesson: out_proj+residual fusion is blocked by the 4D tensor layout
The 14b model's UP_DOWN fusion trick works because it uses flat 2D tensors. Our 4D chunked
layout encodes batch in dimension 2 (not 0), which breaks the UB split. To port this
optimization, the entire tensor layout would need to change from 4D chunked to 2D flat —
a fundamental redesign out of scope for this experiment.

## down_proj DOWN_N_CHUNK 512->256
- Result: 2159 vs 2021, big regression. Total exec 5915->9056 core-us, exec% 98->77%.
  The cube strongly prefers large N; smaller N tiles are inefficient for M=16 decode GEMMs.

## Various spmd replacements for parallel+at
- All current tasks have tiny head OH (0.5-9us except gate/up which are bandwidth-queue-bound).
  Converting to spmd wouldn't change gate/up's 60us HBM queue wait or down_proj's 0.56us head OH.
  Not worth attempting.

## gate+up+silu fused in one pl.at block
- Result: Mat buffer OOM 557KB > 512KB. Two interleaved B-tile ping-pong pairs (wg+wu) in
  the same pipeline stage accumulate 2×[256,256]=2×128KB + stage=2 overhead = 557KB.
- Lesson: At K_CHUNK=256 the B-tile is [256,256]=128KB. With TWO tiles in a merged kernel
  the effective B-stage budget is 2×128KB=256KB, and stage=2 ping-pong doubles that to 512KB —
  right at the limit. The auto_chunk windowing overhead pushes it over. Gate and up must
  remain in separate tasks. (Non-4d confirmed this at MLP_OUT_CHUNK=512; 256 hits the same wall.)

## down_proj K pipeline stage 2->3
- Result: 1955 vs 1944us (regression, high std). down_proj is 99% exec-bound; deeper pipeline
  adds buffer management overhead that outweighs any prefetch benefit.

## gate/up K pipeline stage 2->3
- Result: 1946 vs 1944us (flat). gate/up head OH is 60us HBM bandwidth queue — pipeline
  depth doesn't change queuing wait time.

## k/v_proj + q_proj all stage=3 (q_proj additional step)
- q_proj stage=3 alone: marginal (avg ~1944 vs ~1945 with just k/v stage=3). Negligible.

## Out_proj K pipeline stage=4
- Not tried: stage=4 would need 4×128KB=512KB Mat buffer, exactly at limit. No headroom.

## gate+up+silu fused with pl.range K-loop (no ping-pong)
- Result: 2174 vs 1944, massive regression. gate_proj exec 75->177us (2.4x slower).
- Lesson: Removing ping-pong (pl.range vs pl.pipeline) triples the exec time by eliminating
  HBM prefetch overlap. The 100 silu task savings cannot compensate for 2.4x slower per-task
  execution. pl.pipeline is essential for bandwidth-bound matmuls; pl.range is only for tiny ops.

## mlp_tile layout change [100,1,16,256] -> [200,1,16,128] (down_proj index simplification)
- Not tried: would double silu task count (100->200) and complicate assembly, likely regression.
  down_proj already handles the index calculation efficiently.

## attn_proj_tile chunk doubling (OUT_PROJ_K_CHUNK 128->256)
- Not tried: requires changing how online_softmax assembles attn_proj_tile (ctx [8,128] ->
  grouped into 256-element chunks), complex surgery with no guaranteed win.
  B-tile is already at maximum ([128,512]=128KB, stage=3=384KB near limit).
