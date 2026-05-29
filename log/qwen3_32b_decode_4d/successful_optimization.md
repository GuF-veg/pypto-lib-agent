# Successful Optimizations (qwen3_32b_decode_4d)

Baseline: 2072.91 us (commit 7662590, unmodified, 100 repeat / 5 warmup).

The 4d variant is a fresh starting point: it uses 4D chunked tensor layouts and
has NO split-K applied. The non-4d sibling (`qwen3_32b_decode.py`) was optimized
from ~1999 to ~1789 us; its `log/qwen3_32b_decode/successful_optimization.md`
records that the dominant lever was split-K for the K-dominant decode GEMMs.
The kernel is HBM-bandwidth bound (~1.56 GB of bf16 weights).

## 1. Enlarge down_proj K chunk (DOWN_K_CHUNK 64 -> 128)

- **Result**: 2072.91 -> ~2051 us (2050.80 / 2050.97 across two runs), ~1.1% faster, PASS.
- **Rationale**: down_proj is the single most expensive task (388 us/task * 16 = ~6.2 ms
  total exec, 98.6% exec%). Its K reduction ran over DOWN_K_BLOCKS = INTERMEDIATE/64 = 400
  pipeline iterations, each a tiny [16,64]x[64,512] matmul. Doubling DOWN_K_CHUNK to 128
  halves the iterations (400 -> 200) and each matmul reduces 128 instead of 64 elements,
  cutting per-iteration loop/sync overhead. down_proj exec dropped 388 -> 369 us.
- **Note**: DOWN_K_CHUNK=256 OOMs the Mat buffer (540672 B > 524288 B platform limit) —
  the [16,256]x[256,512] matmul is too large. 128 is the working ceiling for this matmul.
- **Why it's a clean win**: single-constant change, no added complexity.

## 2. q_proj output chunk decoupled to 512 (Q_PROJ_OUT_CHUNK)

- **Result**: ~2051 -> ~2021 us, ~1.4% faster, PASS.
- **Rationale**: q_proj used the shared Q_OUT_CHUNK=256 -> 32 tasks at only 62.8% exec%
  (40 us head OH each, queueing on the cube). Q_OUT_CHUNK cannot simply be raised because
  the golden reference uses it as the out_proj N-half split. Introducing a separate
  Q_PROJ_OUT_CHUNK=512 (q_proj/wq/attention-q-indexing only) makes q_proj spawn 16 larger
  tasks at 89.1% exec%, head OH ~0.6 us. Verified consistent with golden (wq reshape is
  layout-agnostic; attention head indexing recomputed from the new chunk).
- **Why it's a clean win**: mirrors the proven non-4d optimization; localized constant +
  indexing change.

## 3. KV_OUT_CHUNK 256 -> 512

- **Result**: ~2021 -> ~2016 us (2013.69 / 2018.40), ~0.3% faster (near noise), PASS.
- **Rationale**: k_proj and v_proj each had 4 tasks (KV_HIDDEN/256). Doubling the chunk to
  512 halves them to 2 tasks each at 96% exec%, shaving a little dispatch/head OH. In 4d k
  and v are separate `pl.at` scopes (one accumulator each), so unlike the non-4d sibling
  this does NOT OOM (non-4d held k_acc+v_acc together). Marginal but clean and consistent.

## 4. K_CHUNK 128 -> 256 (MLP gate/up + post_rmsnorm K granularity)

- **Result**: ~2016 -> ~1972 us (1978.04 / 1967.28), ~2.2% faster, PASS.
- **Rationale**: K_CHUNK is the K-reduction granularity for the gate/up projections AND for
  post_rmsnorm. Doubling it to 256 halves the pipeline iteration count (K_BLOCKS 64 -> 32).
  The decisive effect is on post_rmsnorm: it is a single serializing task between
  out_proj_residual and gate/up, and at K_CHUNK=128 it had a ~63 us tail-OH bubble (the
  scheduler was slow to detect its completion). Halving its iteration count cut exec 32->24 us
  and the tail OH 63 -> 12 us, removing most of that bubble from the critical path. gate/up
  per-task exec also dipped slightly. Verified consistent with golden (w_gate/w_up reshape
  and post_rms_weight reshape are layout-agnostic).
- **Note**: surprising given gate/up are HBM-bandwidth bound — the win is the post_rmsnorm
  critical-path bubble, not the matmul throughput. Mat buffer stays within limit
  ([16,256]x[256,256] gate/up tiles).

## 5. out_proj single full-width matmul (remove lo/hi 256 split)

- **Result**: ~1972 -> ~1966 us, ~0.3% faster + simpler code, PASS.
- **Rationale**: out_proj split each 512-wide output block into two 256 halves (o_acc_lo /
  o_acc_hi), doing two [16,128]x[128,256] matmuls per K-step. But down_proj already proves a
  single [16,128]x[128,512] matmul fits the Mat buffer (weight tile 128KB, ping-pong 256KB,
  acc [16,512] 32KB ~= 290KB < 512KB). Collapsing to one full-width matmul lets the cube work
  on a larger N tile: out_proj exec dropped 152.8 -> 136.0 us/task. The Q_OUT_CHUNK lo/hi
  split was unnecessary here (the residual pass reads out_proj_tile in 128-chunks regardless).
- **Why it's a clean win**: removes code AND speeds the kernel — a simplification win.

## 6. Decouple post_rmsnorm K chunk to 512 (POST_RMS_K_CHUNK)

- **Result**: ~1966 -> ~1954 us (1950.28 / 1958.69), ~0.6% faster + simpler code, PASS.
- **Rationale**: post_rmsnorm is a vector-only op (no cube Mat buffer). Its K granularity
  was tied to K_CHUNK=256 (32 loop iterations over HIDDEN=8192). Introducing POST_RMS_K_CHUNK=512
  = OUT_PROJ_N_CHUNK makes each resid1_tile block map 1:1 to one iteration (no block/offset
  arithmetic), halving the loop count to 16. post_rmsnorm exec dropped 23.3->18.8 us and its
  tail-OH bubble shrank further. The gate/up K-loops still use K_CHUNK=256 (needed for the
  cube Mat buffer limit). Also simplifies the loop body: removes the resid_block/resid_offset
  index calculations that were needed to navigate the 256-chunk grid over 512-wide resid1_tile.
- **Why it's a clean win**: reduces code complexity AND cuts the critical-path bubble.

## 7. out_proj K pipeline stage 2->3

- **Result**: ~1950 -> ~1950 us avg (marginal, within noise). out_proj exec% 96.3->97.2%, 3 runs avg ~1950.8 us. PASS.
- **Rationale**: OUT_PROJ_K_BLOCKS=64 iterations with B-tile [128,512]=128KB. stage=3 uses 3×128KB=384KB
  Mat buffer (well within 512KB limit). Deeper pipeline overlaps 2 pending HBM fetches instead of 1.
  Marginally better exec% and small trend improvement vs stage=2 with no added complexity.
- **Note**: improvement is within noise (std ~25us across 100 reps). Kept because exec% improved
  and it never regressed in 3 repeated runs.

## 8. k/v_proj K pipeline stage 2->3

- **Result**: ~1951 -> ~1945 us (1945.27 / 1944.49), ~0.3% faster, PASS.
- **Rationale**: k_proj and v_proj each have HIDDEN_K_BLOCKS=64 iterations with B-tile
  [128,512]=128KB bf16. stage=3 uses 3×128KB=384KB Mat buffer (within 512KB limit), overlapping
  2 HBM fetches. exec% improved ~97→98%. Same pattern as out_proj stage 2→3 win.
- **Why it's a clean win**: single-constant change, mirrors out_proj improvement.
