# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 dynamic prefill compressor for the ratio-4 overlapping KV cache.

Copied from models/deepseek_v4_flash_dspark/prefill_compressor_ratio4.py

EXAM: `_prefill_compressor_ratio4_tile` is the exercise -- its body is left
unimplemented. `golden_prefill_compressor_ratio4` is the torch reference for the
required numerics and `prefill_compressor_ratio4_test` is the harness entry that
validates against it.
"""


import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "golden" / "__init__.py").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import torch
import pypto.language as pl

from golden import TensorSpec, ratio_allclose, run


# Dynamic shape variables.
T_DYN = pl.dynamic("PREFILL_CSA_C4_T_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
QUERY_START_LOC_DYN = pl.dynamic("PREFILL_METADATA_QUERY_START_LOC_DYN")
REQUESTS_DYN = pl.dynamic("PREFILL_METADATA_REQUESTS_DYN")

# model config (FLASH preset of config.py)
EPS = 1e-6
D = 4096
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = 1_048_576
START_POS = 0
COMPRESS_RATIO = 4
OUT_DIM = 2 * HEAD_DIM
STATE_LEN = 2 * COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM

# deployment (PREFILL_SEQ of config.py)
PREFILL_SEQ = 512

# paged compressed state / KV cache (BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE,
# CSA_STATE_BLOCKS_PER_REQUEST, CSA_STATE_PHYSICAL_BLOCKS of config.py)
BLOCK_SIZE = 32
CSA_STATE_BLOCK_SIZE = 2
CSA_STATE_BLOCKS_PER_REQUEST = 8
CSA_STATE_BLOCK_NUM = 512
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE

# Bounded physical-row tile for ratio-4 projection/state updates.
PREFILL_STATE_TILE = 512

# write-row tile of the norm/rope stage
PACKED_RMS_TILE = 16

assert PREFILL_STATE_TILE % COMPRESS_RATIO == 0


# model config (FLASH preset of config.py, only the fields the RoPE fixture reads)
@dataclass(frozen=True)
class DeepSeekV4RoPEConfig:
    qk_rope_head_dim: int
    compress_rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_max_position_embeddings: int


M = DeepSeekV4RoPEConfig(
    qk_rope_head_dim=64,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
)


@pl.jit.inline(auto_scope=False)
def _c4_proj(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w: pl.Tensor[[OUT_DIM, D], pl.BF16],
    t: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
):
    acc = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    for kb in pl.range(0, D, 256):
        k0 = kb * 32
        acc = pl.matmul_acc(
            acc,
            x[t : t + 16, k0 : k0 + 32],
            w[col : col + HEAD_DIM, k0 : k0 + 32],
            b_trans=True,
            init_cond=(kb == 0),
        )
    return acc


@pl.jit.inline(auto_scope=False)
def _prefill_compressor_ratio4_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    rope_dup_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_swap_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_sign_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM], pl.FP32],
    rope_even_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], pl.INT32],
    rope_odd_idx_template: pl.Tensor[[PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], pl.INT32],
    state_order_fence: pl.InOut[pl.Tensor[[1], pl.INT32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Pool the ratio-4 window, normalize, rotate and publish one 512-row state tile."""
    base = tile_base
    rows = tile_rows
    start = START_POS
    pre_ak = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    pre_as = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    pre_bk = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    pre_bs = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    apeA = pl.create_tensor([COMPRESS_RATIO, HEAD_DIM], dtype=pl.FP32)
    apeB = pl.create_tensor([COMPRESS_RATIO, HEAD_DIM], dtype=pl.FP32)
    pub_kv = pl.create_tensor([16, OUT_DIM], dtype=pl.FP32)
    pub_sc = pl.create_tensor([16, OUT_DIM], dtype=pl.FP32)
    bbf_kv = pl.create_tensor([16, OUT_DIM], dtype=pl.FP32)
    bbf_sc = pl.create_tensor([16, OUT_DIM], dtype=pl.FP32)
    rotbuf = pl.create_tensor([1, ROPE_HEAD_DIM], dtype=pl.FP32)
    sq16 = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    nt16 = pl.create_tensor([16, 1, HEAD_DIM], dtype=pl.FP32)
    apeA16 = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    apeB16 = pl.create_tensor([16, HEAD_DIM], dtype=pl.FP32)
    nt = pl.create_tensor([1, HEAD_DIM], dtype=pl.FP32)
    sqpad = pl.create_tensor([8, HEAD_DIM], dtype=pl.FP32)
    invbuf = pl.create_tensor([8, 1], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_c4_tile",
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ):
        zero_row = pl.mul(pl.load(ape, [0, 0], [1, OUT_DIM]), 0.0)
        neg_row = pl.add(zero_row, -1.0e30)
        zero_head = pl.mul(pl.load(ape, [0, 0], [1, HEAD_DIM]), 0.0)
        neg_head = pl.add(zero_head, -1.0e30)

        # Raw projections of the STATE_LEN tokens immediately before this tile.
        for s in pl.range(STATE_LEN):
            ppos = start + base - STATE_LEN + s
            if ppos >= 0:
                pblock = ppos // CSA_STATE_BLOCK_SIZE
                pintra = ppos % CSA_STATE_BLOCK_SIZE
                pphys = pl.read(compress_state_block_table, [pblock])
                p_kv_a = pl.reshape(pl.load(compress_state, [pphys, pintra, 0], [1, 1, HEAD_DIM]), [1, HEAD_DIM])
                p_sc_a = pl.reshape(pl.load(compress_state, [pphys, pintra, OUT_DIM], [1, 1, HEAD_DIM]), [1, HEAD_DIM])
                p_kv_b = pl.reshape(pl.load(compress_state, [pphys, pintra, HEAD_DIM], [1, 1, HEAD_DIM]), [1, HEAD_DIM])
                p_sc_b = pl.reshape(pl.load(compress_state, [pphys, pintra, OUT_DIM + HEAD_DIM], [1, 1, HEAD_DIM]), [1, HEAD_DIM])
                p_ape_a = pl.load(ape, [ppos % COMPRESS_RATIO, 0], [1, HEAD_DIM])
                p_ape_b = pl.load(ape, [ppos % COMPRESS_RATIO, HEAD_DIM], [1, HEAD_DIM])
                pl.store(p_kv_a, [s, 0], pre_ak)
                pl.store(pl.sub(p_sc_a, p_ape_a), [s, 0], pre_as)
                pl.store(p_kv_b, [s, 0], pre_bk)
                pl.store(pl.sub(p_sc_b, p_ape_b), [s, 0], pre_bs)
            else:
                pl.store(zero_head, [s, 0], pre_ak)
                pl.store(neg_head, [s, 0], pre_as)
                pl.store(zero_head, [s, 0], pre_bk)
                pl.store(neg_head, [s, 0], pre_bs)
        rsum_tmp = pl.tile.create([8, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        ctmp = pl.tile.create([16, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        ctmp16 = pl.tile.create([16, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        for rp in pl.range(8):
            pl.store(zero_head, [rp, 0], sqpad)
        pl.store(pl.load(ape, [0, 0], [COMPRESS_RATIO, HEAD_DIM]), [0, 0], apeA)
        pl.store(pl.load(ape, [0, HEAD_DIM], [COMPRESS_RATIO, HEAD_DIM]), [0, 0], apeB)
        even_idx = rope_even_idx_template[0:1, 0:ROPE_HEAD_DIM // 2]
        odd_idx = rope_odd_idx_template[0:1, 0:ROPE_HEAD_DIM // 2]

        # One pool window per boundary token. Group A is the previous ratio-4 span,
        # group B the current one; each spans four slots drawn from the projection
        # window or, before the sequence start, from the published state prefix.
        for k in pl.range(0, rows):
            tid = base + k
            dst_row = pl.read(cmp_slot_mapping, [tid])
            if dst_row >= 0:
                wa = pl.max(tid - 15, 0)
                wA = pl.max(tid, STATE_LEN - 1) - (STATE_LEN - 1)
                wB = pl.max(tid, COMPRESS_RATIO - 1) - (COMPRESS_RATIO - 1)
                nrA = pl.min(pl.max(tid, COMPRESS_RATIO - 1) - (COMPRESS_RATIO - 1), COMPRESS_RATIO)
                nrB = pl.min(tid + 1, COMPRESS_RATIO)
                nvA = COMPRESS_RATIO - nrA
                nvB = COMPRESS_RATIO - nrB
                pA = pl.min(tid + 1, 8)
                pB = pl.min(tid + 5, 8)
                wpos = pl.cast(pl.read(position_ids, [tid]), pl.FP32)
                posf = pl.cast(pl.load(position_ids, [wa], [16]), pl.FP32)
                aA = pl.maximum(pl.sub(posf, pl.sub(wpos, 4.0)), 0.0)
                bA = pl.maximum(pl.mul(pl.sub(posf, pl.sub(wpos, 7.0)), -1.0), 0.0)
                aB = pl.maximum(pl.sub(posf, wpos), 0.0)
                bB = pl.maximum(pl.mul(pl.sub(posf, pl.sub(wpos, 3.0)), -1.0), 0.0)
                m1A = pl.reshape(pl.add(pl.mul(pl.minimum(pl.add(aA, bA), 1.0), -1.0), 1.0), [16, 1])
                wpos_i = pl.read(position_ids, [tid])
                gateA = pl.cast(pl.cast(pl.min(pl.max(pl.sub(wpos_i, 6), 0), 1), pl.INT32), pl.FP32)
                m1A = pl.mul(m1A, gateA)
                m1B = pl.reshape(pl.add(pl.mul(pl.minimum(pl.add(aB, bB), 1.0), -1.0), 1.0), [16, 1])
                kva = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                sca = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                kvb = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                scb = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                xw = pl.load(x, [wa, 0], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul(xw, pl.tile.transpose_view(pl.load(wkv, [0, 0], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul(xw, pl.tile.transpose_view(pl.load(wgate, [0, 0], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul(xw, pl.tile.transpose_view(pl.load(wkv, [512, 0], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul(xw, pl.tile.transpose_view(pl.load(wgate, [512, 0], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 512], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul_acc(kva, xw, pl.tile.transpose_view(pl.load(wkv, [0, 512], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul_acc(sca, xw, pl.tile.transpose_view(pl.load(wgate, [0, 512], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul_acc(kvb, xw, pl.tile.transpose_view(pl.load(wkv, [512, 512], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul_acc(scb, xw, pl.tile.transpose_view(pl.load(wgate, [512, 512], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 1024], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul_acc(kva, xw, pl.tile.transpose_view(pl.load(wkv, [0, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul_acc(sca, xw, pl.tile.transpose_view(pl.load(wgate, [0, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul_acc(kvb, xw, pl.tile.transpose_view(pl.load(wkv, [512, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul_acc(scb, xw, pl.tile.transpose_view(pl.load(wgate, [512, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 1536], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul_acc(kva, xw, pl.tile.transpose_view(pl.load(wkv, [0, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul_acc(sca, xw, pl.tile.transpose_view(pl.load(wgate, [0, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul_acc(kvb, xw, pl.tile.transpose_view(pl.load(wkv, [512, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul_acc(scb, xw, pl.tile.transpose_view(pl.load(wgate, [512, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 2048], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul_acc(kva, xw, pl.tile.transpose_view(pl.load(wkv, [0, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul_acc(sca, xw, pl.tile.transpose_view(pl.load(wgate, [0, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul_acc(kvb, xw, pl.tile.transpose_view(pl.load(wkv, [512, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul_acc(scb, xw, pl.tile.transpose_view(pl.load(wgate, [512, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 2560], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul_acc(kva, xw, pl.tile.transpose_view(pl.load(wkv, [0, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul_acc(sca, xw, pl.tile.transpose_view(pl.load(wgate, [0, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul_acc(kvb, xw, pl.tile.transpose_view(pl.load(wkv, [512, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul_acc(scb, xw, pl.tile.transpose_view(pl.load(wgate, [512, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 3072], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul_acc(kva, xw, pl.tile.transpose_view(pl.load(wkv, [0, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul_acc(sca, xw, pl.tile.transpose_view(pl.load(wgate, [0, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul_acc(kvb, xw, pl.tile.transpose_view(pl.load(wkv, [512, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul_acc(scb, xw, pl.tile.transpose_view(pl.load(wgate, [512, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 3584], [16, 512], target_memory=pl.Mem.Mat)
                kva = pl.matmul_acc(kva, xw, pl.tile.transpose_view(pl.load(wkv, [0, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                sca = pl.matmul_acc(sca, xw, pl.tile.transpose_view(pl.load(wgate, [0, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                kvb = pl.matmul_acc(kvb, xw, pl.tile.transpose_view(pl.load(wkv, [512, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                scb = pl.matmul_acc(scb, xw, pl.tile.transpose_view(pl.load(wgate, [512, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                A0_0 = pl.load(ape, [(pl.read(position_ids, [wa + 0]) % COMPRESS_RATIO), 0], [1, 256])
                A1_0 = pl.load(ape, [(pl.read(position_ids, [wa + 1]) % COMPRESS_RATIO), 0], [1, 256])
                A2_0 = pl.load(ape, [(pl.read(position_ids, [wa + 2]) % COMPRESS_RATIO), 0], [1, 256])
                A3_0 = pl.load(ape, [(pl.read(position_ids, [wa + 3]) % COMPRESS_RATIO), 0], [1, 256])
                rA_0 = pl.concat(pl.concat(pl.concat(A0_0, A1_0), A2_0), A3_0)
                apeWA0 = pl.reshape(pl.concat(pl.concat(pl.concat(rA_0, rA_0), rA_0), rA_0), [16, 256])
                B0_0 = pl.load(ape, [(pl.read(position_ids, [wa + 0]) % COMPRESS_RATIO), 512], [1, 256])
                B1_0 = pl.load(ape, [(pl.read(position_ids, [wa + 1]) % COMPRESS_RATIO), 512], [1, 256])
                B2_0 = pl.load(ape, [(pl.read(position_ids, [wa + 2]) % COMPRESS_RATIO), 512], [1, 256])
                B3_0 = pl.load(ape, [(pl.read(position_ids, [wa + 3]) % COMPRESS_RATIO), 512], [1, 256])
                rB_0 = pl.concat(pl.concat(pl.concat(B0_0, B1_0), B2_0), B3_0)
                apeWB0 = pl.reshape(pl.concat(pl.concat(pl.concat(rB_0, rB_0), rB_0), rB_0), [16, 256])
                scA0 = pl.minimum(pl.add(sca, apeWA0), 20.0)
                scB0 = pl.minimum(pl.add(scb, apeWB0), 20.0)
                wA0 = pl.mul(pl.exp(scA0), pl.row_expand_mul(pl.add(pl.mul(sca, 0.0), 1.0), m1A))
                wB0 = pl.mul(pl.exp(scB0), pl.row_expand_mul(pl.add(pl.mul(scb, 0.0), 1.0), m1B))
                den0 = pl.add(pl.col_sum(wA0), pl.col_sum(wB0))
                num0 = pl.add(pl.col_sum(pl.mul(wA0, kva)), pl.col_sum(pl.mul(wB0, kvb)))
                kk = tid - base
                gAI = pl.min(pl.max(pl.sub(wpos_i, 6), 0), 1)
                pposA0 = wpos_i - 7 + 0
                if tid + 0 < 7:
                    if gAI >= 1:
                        if pposA0 >= 0:
                            pblkA0 = pposA0 // CSA_STATE_BLOCK_SIZE
                            pintA0 = pposA0 % CSA_STATE_BLOCK_SIZE
                            pphA0 = pl.read(compress_state_block_table, [pblkA0])
                            if pphA0 >= 0:
                                skvA0 = pl.reshape(pl.load(compress_state, [pphA0, pintA0, 0], [1, 1, 256]), [1, 256])
                                sscA0 = pl.reshape(pl.load(compress_state, [pphA0, pintA0, OUT_DIM + 0], [1, 1, 256]), [1, 256])
                                wgtA0 = pl.exp(sscA0)
                                num0 = pl.add(num0, pl.mul(wgtA0, skvA0))
                                den0 = pl.add(den0, wgtA0)
                pposA1 = wpos_i - 7 + 1
                if tid + 1 < 7:
                    if gAI >= 1:
                        if pposA1 >= 0:
                            pblkA1 = pposA1 // CSA_STATE_BLOCK_SIZE
                            pintA1 = pposA1 % CSA_STATE_BLOCK_SIZE
                            pphA1 = pl.read(compress_state_block_table, [pblkA1])
                            if pphA1 >= 0:
                                skvA1 = pl.reshape(pl.load(compress_state, [pphA1, pintA1, 0], [1, 1, 256]), [1, 256])
                                sscA1 = pl.reshape(pl.load(compress_state, [pphA1, pintA1, OUT_DIM + 0], [1, 1, 256]), [1, 256])
                                wgtA1 = pl.exp(sscA1)
                                num0 = pl.add(num0, pl.mul(wgtA1, skvA1))
                                den0 = pl.add(den0, wgtA1)
                pposA2 = wpos_i - 7 + 2
                if tid + 2 < 7:
                    if gAI >= 1:
                        if pposA2 >= 0:
                            pblkA2 = pposA2 // CSA_STATE_BLOCK_SIZE
                            pintA2 = pposA2 % CSA_STATE_BLOCK_SIZE
                            pphA2 = pl.read(compress_state_block_table, [pblkA2])
                            if pphA2 >= 0:
                                skvA2 = pl.reshape(pl.load(compress_state, [pphA2, pintA2, 0], [1, 1, 256]), [1, 256])
                                sscA2 = pl.reshape(pl.load(compress_state, [pphA2, pintA2, OUT_DIM + 0], [1, 1, 256]), [1, 256])
                                wgtA2 = pl.exp(sscA2)
                                num0 = pl.add(num0, pl.mul(wgtA2, skvA2))
                                den0 = pl.add(den0, wgtA2)
                pposA3 = wpos_i - 7 + 3
                if tid + 3 < 7:
                    if gAI >= 1:
                        if pposA3 >= 0:
                            pblkA3 = pposA3 // CSA_STATE_BLOCK_SIZE
                            pintA3 = pposA3 % CSA_STATE_BLOCK_SIZE
                            pphA3 = pl.read(compress_state_block_table, [pblkA3])
                            if pphA3 >= 0:
                                skvA3 = pl.reshape(pl.load(compress_state, [pphA3, pintA3, 0], [1, 1, 256]), [1, 256])
                                sscA3 = pl.reshape(pl.load(compress_state, [pphA3, pintA3, OUT_DIM + 0], [1, 1, 256]), [1, 256])
                                wgtA3 = pl.exp(sscA3)
                                num0 = pl.add(num0, pl.mul(wgtA3, skvA3))
                                den0 = pl.add(den0, wgtA3)
                pposB0 = wpos_i - 3 + 0
                if tid + 0 < 3:
                    if pposB0 >= 0:
                        pblkB0 = pposB0 // CSA_STATE_BLOCK_SIZE
                        pintB0 = pposB0 % CSA_STATE_BLOCK_SIZE
                        pphB0 = pl.read(compress_state_block_table, [pblkB0])
                        if pphB0 >= 0:
                            skvB0 = pl.reshape(pl.load(compress_state, [pphB0, pintB0, HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            sscB0 = pl.reshape(pl.load(compress_state, [pphB0, pintB0, OUT_DIM + HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            wgtB0 = pl.exp(sscB0)
                            num0 = pl.add(num0, pl.mul(wgtB0, skvB0))
                            den0 = pl.add(den0, wgtB0)
                pposB1 = wpos_i - 3 + 1
                if tid + 1 < 3:
                    if pposB1 >= 0:
                        pblkB1 = pposB1 // CSA_STATE_BLOCK_SIZE
                        pintB1 = pposB1 % CSA_STATE_BLOCK_SIZE
                        pphB1 = pl.read(compress_state_block_table, [pblkB1])
                        if pphB1 >= 0:
                            skvB1 = pl.reshape(pl.load(compress_state, [pphB1, pintB1, HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            sscB1 = pl.reshape(pl.load(compress_state, [pphB1, pintB1, OUT_DIM + HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            wgtB1 = pl.exp(sscB1)
                            num0 = pl.add(num0, pl.mul(wgtB1, skvB1))
                            den0 = pl.add(den0, wgtB1)
                pposB2 = wpos_i - 3 + 2
                if tid + 2 < 3:
                    if pposB2 >= 0:
                        pblkB2 = pposB2 // CSA_STATE_BLOCK_SIZE
                        pintB2 = pposB2 % CSA_STATE_BLOCK_SIZE
                        pphB2 = pl.read(compress_state_block_table, [pblkB2])
                        if pphB2 >= 0:
                            skvB2 = pl.reshape(pl.load(compress_state, [pphB2, pintB2, HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            sscB2 = pl.reshape(pl.load(compress_state, [pphB2, pintB2, OUT_DIM + HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            wgtB2 = pl.exp(sscB2)
                            num0 = pl.add(num0, pl.mul(wgtB2, skvB2))
                            den0 = pl.add(den0, wgtB2)
                pposB3 = wpos_i - 3 + 3
                if tid + 3 < 3:
                    if pposB3 >= 0:
                        pblkB3 = pposB3 // CSA_STATE_BLOCK_SIZE
                        pintB3 = pposB3 % CSA_STATE_BLOCK_SIZE
                        pphB3 = pl.read(compress_state_block_table, [pblkB3])
                        if pphB3 >= 0:
                            skvB3 = pl.reshape(pl.load(compress_state, [pphB3, pintB3, HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            sscB3 = pl.reshape(pl.load(compress_state, [pphB3, pintB3, OUT_DIM + HEAD_DIM + 0], [1, 1, 256]), [1, 256])
                            wgtB3 = pl.exp(sscB3)
                            num0 = pl.add(num0, pl.mul(wgtB3, skvB3))
                            den0 = pl.add(den0, wgtB3)
                p0 = pl.div(num0, pl.add(den0, EPS))
                kva1 = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                sca1 = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                kvb1 = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                scb1 = pl.create_tile([16, 256], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                xw = pl.load(x, [wa, 0], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul(xw, pl.tile.transpose_view(pl.load(wkv, [256, 0], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul(xw, pl.tile.transpose_view(pl.load(wgate, [256, 0], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul(xw, pl.tile.transpose_view(pl.load(wkv, [768, 0], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul(xw, pl.tile.transpose_view(pl.load(wgate, [768, 0], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 512], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul_acc(kva1, xw, pl.tile.transpose_view(pl.load(wkv, [256, 512], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul_acc(sca1, xw, pl.tile.transpose_view(pl.load(wgate, [256, 512], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul_acc(kvb1, xw, pl.tile.transpose_view(pl.load(wkv, [768, 512], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul_acc(scb1, xw, pl.tile.transpose_view(pl.load(wgate, [768, 512], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 1024], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul_acc(kva1, xw, pl.tile.transpose_view(pl.load(wkv, [256, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul_acc(sca1, xw, pl.tile.transpose_view(pl.load(wgate, [256, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul_acc(kvb1, xw, pl.tile.transpose_view(pl.load(wkv, [768, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul_acc(scb1, xw, pl.tile.transpose_view(pl.load(wgate, [768, 1024], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 1536], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul_acc(kva1, xw, pl.tile.transpose_view(pl.load(wkv, [256, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul_acc(sca1, xw, pl.tile.transpose_view(pl.load(wgate, [256, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul_acc(kvb1, xw, pl.tile.transpose_view(pl.load(wkv, [768, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul_acc(scb1, xw, pl.tile.transpose_view(pl.load(wgate, [768, 1536], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 2048], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul_acc(kva1, xw, pl.tile.transpose_view(pl.load(wkv, [256, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul_acc(sca1, xw, pl.tile.transpose_view(pl.load(wgate, [256, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul_acc(kvb1, xw, pl.tile.transpose_view(pl.load(wkv, [768, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul_acc(scb1, xw, pl.tile.transpose_view(pl.load(wgate, [768, 2048], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 2560], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul_acc(kva1, xw, pl.tile.transpose_view(pl.load(wkv, [256, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul_acc(sca1, xw, pl.tile.transpose_view(pl.load(wgate, [256, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul_acc(kvb1, xw, pl.tile.transpose_view(pl.load(wkv, [768, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul_acc(scb1, xw, pl.tile.transpose_view(pl.load(wgate, [768, 2560], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 3072], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul_acc(kva1, xw, pl.tile.transpose_view(pl.load(wkv, [256, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul_acc(sca1, xw, pl.tile.transpose_view(pl.load(wgate, [256, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul_acc(kvb1, xw, pl.tile.transpose_view(pl.load(wkv, [768, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul_acc(scb1, xw, pl.tile.transpose_view(pl.load(wgate, [768, 3072], [256, 512], target_memory=pl.Mem.Mat)))
                xw = pl.load(x, [wa, 3584], [16, 512], target_memory=pl.Mem.Mat)
                kva1 = pl.matmul_acc(kva1, xw, pl.tile.transpose_view(pl.load(wkv, [256, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                sca1 = pl.matmul_acc(sca1, xw, pl.tile.transpose_view(pl.load(wgate, [256, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                kvb1 = pl.matmul_acc(kvb1, xw, pl.tile.transpose_view(pl.load(wkv, [768, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                scb1 = pl.matmul_acc(scb1, xw, pl.tile.transpose_view(pl.load(wgate, [768, 3584], [256, 512], target_memory=pl.Mem.Mat)))
                A0_1 = pl.load(ape, [(pl.read(position_ids, [wa + 0]) % COMPRESS_RATIO), 256], [1, 256])
                A1_1 = pl.load(ape, [(pl.read(position_ids, [wa + 1]) % COMPRESS_RATIO), 256], [1, 256])
                A2_1 = pl.load(ape, [(pl.read(position_ids, [wa + 2]) % COMPRESS_RATIO), 256], [1, 256])
                A3_1 = pl.load(ape, [(pl.read(position_ids, [wa + 3]) % COMPRESS_RATIO), 256], [1, 256])
                rA_1 = pl.concat(pl.concat(pl.concat(A0_1, A1_1), A2_1), A3_1)
                apeWA1 = pl.reshape(pl.concat(pl.concat(pl.concat(rA_1, rA_1), rA_1), rA_1), [16, 256])
                B0_1 = pl.load(ape, [(pl.read(position_ids, [wa + 0]) % COMPRESS_RATIO), 768], [1, 256])
                B1_1 = pl.load(ape, [(pl.read(position_ids, [wa + 1]) % COMPRESS_RATIO), 768], [1, 256])
                B2_1 = pl.load(ape, [(pl.read(position_ids, [wa + 2]) % COMPRESS_RATIO), 768], [1, 256])
                B3_1 = pl.load(ape, [(pl.read(position_ids, [wa + 3]) % COMPRESS_RATIO), 768], [1, 256])
                rB_1 = pl.concat(pl.concat(pl.concat(B0_1, B1_1), B2_1), B3_1)
                apeWB1 = pl.reshape(pl.concat(pl.concat(pl.concat(rB_1, rB_1), rB_1), rB_1), [16, 256])
                scA1 = pl.minimum(pl.add(sca1, apeWA1), 20.0)
                scB1 = pl.minimum(pl.add(scb1, apeWB1), 20.0)
                wA1 = pl.mul(pl.exp(scA1), pl.row_expand_mul(pl.add(pl.mul(sca1, 0.0), 1.0), m1A))
                wB1 = pl.mul(pl.exp(scB1), pl.row_expand_mul(pl.add(pl.mul(scb1, 0.0), 1.0), m1B))
                den1 = pl.add(pl.col_sum(wA1), pl.col_sum(wB1))
                num1 = pl.add(pl.col_sum(pl.mul(wA1, kva1)), pl.col_sum(pl.mul(wB1, kvb1)))
                kk = tid - base
                gAI = pl.min(pl.max(pl.sub(wpos_i, 6), 0), 1)
                pposA0 = wpos_i - 7 + 0
                if tid + 0 < 7:
                    if gAI >= 1:
                        if pposA0 >= 0:
                            pblkA0 = pposA0 // CSA_STATE_BLOCK_SIZE
                            pintA0 = pposA0 % CSA_STATE_BLOCK_SIZE
                            pphA0 = pl.read(compress_state_block_table, [pblkA0])
                            if pphA0 >= 0:
                                skvA0 = pl.reshape(pl.load(compress_state, [pphA0, pintA0, 256], [1, 1, 256]), [1, 256])
                                sscA0 = pl.reshape(pl.load(compress_state, [pphA0, pintA0, OUT_DIM + 256], [1, 1, 256]), [1, 256])
                                wgtA0 = pl.exp(sscA0)
                                num1 = pl.add(num1, pl.mul(wgtA0, skvA0))
                                den1 = pl.add(den1, wgtA0)
                pposA1 = wpos_i - 7 + 1
                if tid + 1 < 7:
                    if gAI >= 1:
                        if pposA1 >= 0:
                            pblkA1 = pposA1 // CSA_STATE_BLOCK_SIZE
                            pintA1 = pposA1 % CSA_STATE_BLOCK_SIZE
                            pphA1 = pl.read(compress_state_block_table, [pblkA1])
                            if pphA1 >= 0:
                                skvA1 = pl.reshape(pl.load(compress_state, [pphA1, pintA1, 256], [1, 1, 256]), [1, 256])
                                sscA1 = pl.reshape(pl.load(compress_state, [pphA1, pintA1, OUT_DIM + 256], [1, 1, 256]), [1, 256])
                                wgtA1 = pl.exp(sscA1)
                                num1 = pl.add(num1, pl.mul(wgtA1, skvA1))
                                den1 = pl.add(den1, wgtA1)
                pposA2 = wpos_i - 7 + 2
                if tid + 2 < 7:
                    if gAI >= 1:
                        if pposA2 >= 0:
                            pblkA2 = pposA2 // CSA_STATE_BLOCK_SIZE
                            pintA2 = pposA2 % CSA_STATE_BLOCK_SIZE
                            pphA2 = pl.read(compress_state_block_table, [pblkA2])
                            if pphA2 >= 0:
                                skvA2 = pl.reshape(pl.load(compress_state, [pphA2, pintA2, 256], [1, 1, 256]), [1, 256])
                                sscA2 = pl.reshape(pl.load(compress_state, [pphA2, pintA2, OUT_DIM + 256], [1, 1, 256]), [1, 256])
                                wgtA2 = pl.exp(sscA2)
                                num1 = pl.add(num1, pl.mul(wgtA2, skvA2))
                                den1 = pl.add(den1, wgtA2)
                pposA3 = wpos_i - 7 + 3
                if tid + 3 < 7:
                    if gAI >= 1:
                        if pposA3 >= 0:
                            pblkA3 = pposA3 // CSA_STATE_BLOCK_SIZE
                            pintA3 = pposA3 % CSA_STATE_BLOCK_SIZE
                            pphA3 = pl.read(compress_state_block_table, [pblkA3])
                            if pphA3 >= 0:
                                skvA3 = pl.reshape(pl.load(compress_state, [pphA3, pintA3, 256], [1, 1, 256]), [1, 256])
                                sscA3 = pl.reshape(pl.load(compress_state, [pphA3, pintA3, OUT_DIM + 256], [1, 1, 256]), [1, 256])
                                wgtA3 = pl.exp(sscA3)
                                num1 = pl.add(num1, pl.mul(wgtA3, skvA3))
                                den1 = pl.add(den1, wgtA3)
                pposB0 = wpos_i - 3 + 0
                if tid + 0 < 3:
                    if pposB0 >= 0:
                        pblkB0 = pposB0 // CSA_STATE_BLOCK_SIZE
                        pintB0 = pposB0 % CSA_STATE_BLOCK_SIZE
                        pphB0 = pl.read(compress_state_block_table, [pblkB0])
                        if pphB0 >= 0:
                            skvB0 = pl.reshape(pl.load(compress_state, [pphB0, pintB0, HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            sscB0 = pl.reshape(pl.load(compress_state, [pphB0, pintB0, OUT_DIM + HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            wgtB0 = pl.exp(sscB0)
                            num1 = pl.add(num1, pl.mul(wgtB0, skvB0))
                            den1 = pl.add(den1, wgtB0)
                pposB1 = wpos_i - 3 + 1
                if tid + 1 < 3:
                    if pposB1 >= 0:
                        pblkB1 = pposB1 // CSA_STATE_BLOCK_SIZE
                        pintB1 = pposB1 % CSA_STATE_BLOCK_SIZE
                        pphB1 = pl.read(compress_state_block_table, [pblkB1])
                        if pphB1 >= 0:
                            skvB1 = pl.reshape(pl.load(compress_state, [pphB1, pintB1, HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            sscB1 = pl.reshape(pl.load(compress_state, [pphB1, pintB1, OUT_DIM + HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            wgtB1 = pl.exp(sscB1)
                            num1 = pl.add(num1, pl.mul(wgtB1, skvB1))
                            den1 = pl.add(den1, wgtB1)
                pposB2 = wpos_i - 3 + 2
                if tid + 2 < 3:
                    if pposB2 >= 0:
                        pblkB2 = pposB2 // CSA_STATE_BLOCK_SIZE
                        pintB2 = pposB2 % CSA_STATE_BLOCK_SIZE
                        pphB2 = pl.read(compress_state_block_table, [pblkB2])
                        if pphB2 >= 0:
                            skvB2 = pl.reshape(pl.load(compress_state, [pphB2, pintB2, HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            sscB2 = pl.reshape(pl.load(compress_state, [pphB2, pintB2, OUT_DIM + HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            wgtB2 = pl.exp(sscB2)
                            num1 = pl.add(num1, pl.mul(wgtB2, skvB2))
                            den1 = pl.add(den1, wgtB2)
                pposB3 = wpos_i - 3 + 3
                if tid + 3 < 3:
                    if pposB3 >= 0:
                        pblkB3 = pposB3 // CSA_STATE_BLOCK_SIZE
                        pintB3 = pposB3 % CSA_STATE_BLOCK_SIZE
                        pphB3 = pl.read(compress_state_block_table, [pblkB3])
                        if pphB3 >= 0:
                            skvB3 = pl.reshape(pl.load(compress_state, [pphB3, pintB3, HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            sscB3 = pl.reshape(pl.load(compress_state, [pphB3, pintB3, OUT_DIM + HEAD_DIM + 256], [1, 1, 256]), [1, 256])
                            wgtB3 = pl.exp(sscB3)
                            num1 = pl.add(num1, pl.mul(wgtB3, skvB3))
                            den1 = pl.add(den1, wgtB3)
                p1 = pl.div(num1, pl.add(den1, EPS))
                q0 = pl.fillpad_expand(p0, [16, 256])
                q1 = pl.fillpad_expand(p1, [16, 256])
                r0 = pl.row_sum(pl.mul(q0, q0), ctmp16)
                r1 = pl.row_sum(pl.mul(q1, q1), ctmp16)
                inv16 = pl.recip(pl.sqrt(pl.add(pl.mul(pl.add(r0, r1), 1.0 / HEAD_DIM), EPS)))
                nw0 = pl.reshape(pl.cast(pl.load(norm_w, [0], [256]), pl.FP32), [1, 256])
                nw1 = pl.reshape(pl.cast(pl.load(norm_w, [256], [256]), pl.FP32), [1, 256])
                pl.store(pl.reshape(pl.col_expand_mul(pl.row_expand_mul(q0, inv16), nw0), [16, 1, 256]), [0, 0, 0], nt16)
                pl.store(pl.reshape(pl.col_expand_mul(pl.row_expand_mul(q1, inv16), nw1), [16, 1, 256]), [0, 0, 256], nt16)
                nt[0:1, 0:HEAD_DIM] = nt16[0:1, 0, 0:HEAD_DIM]
                nrm_t = nt[0:1, 0:HEAD_DIM]
                rope_t = nt[0:1, NOPE_HEAD_DIM:HEAD_DIM]
                cos_h = pl.cast(cmp_freqs_cos[tid : tid + 1, 0 : ROPE_HEAD_DIM // 2], pl.FP32)
                sin_h = pl.cast(cmp_freqs_sin[tid : tid + 1, 0 : ROPE_HEAD_DIM // 2], pl.FP32)
                rswp = rope_swap_idx_template[0:1, 0:ROPE_HEAD_DIM]
                rsgn = rope_sign_template[0:1, 0:ROPE_HEAD_DIM]
                z64 = pl.mul(rope_t, 0.0)
                swp = pl.scatter(z64, dim=-1, index=rswp, src=rope_t)
                cos2 = pl.add(pl.scatter(z64, dim=-1, index=even_idx, src=cos_h), pl.scatter(z64, dim=-1, index=odd_idx, src=cos_h))
                sin2 = pl.add(pl.scatter(z64, dim=-1, index=even_idx, src=sin_h), pl.scatter(z64, dim=-1, index=odd_idx, src=sin_h))
                rot = pl.add(pl.mul(rope_t, cos2), pl.mul(pl.mul(swp, sin2), rsgn))
                rotbuf[0:1, 0:ROPE_HEAD_DIM] = rot
                cblk = dst_row // BLOCK_SIZE
                cint = dst_row % BLOCK_SIZE
                cmp_kv[cblk, cint, 0, 0:NOPE_HEAD_DIM] = pl.reshape(pl.cast(nt[0:1, 0:NOPE_HEAD_DIM], pl.BF16), [NOPE_HEAD_DIM])
                cmp_kv[cblk, cint, 0, NOPE_HEAD_DIM:HEAD_DIM] = pl.reshape(pl.cast(rotbuf, pl.BF16), [ROPE_HEAD_DIM])
        publish_base = base + pl.max(rows, STATE_LEN) - STATE_LEN
        pk0 = pl.create_tile([16, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        pk1 = pl.create_tile([16, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        pxw0 = pl.load(x, [publish_base, 0], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul(pxw0, pl.tile.transpose_view(pl.load(wkv, [0, 0], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul(pxw0, pl.tile.transpose_view(pl.load(wkv, [512, 0], [512, 256], target_memory=pl.Mem.Mat)))
        pxw1 = pl.load(x, [publish_base, 256], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw1, pl.tile.transpose_view(pl.load(wkv, [0, 256], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw1, pl.tile.transpose_view(pl.load(wkv, [512, 256], [512, 256], target_memory=pl.Mem.Mat)))
        pxw2 = pl.load(x, [publish_base, 512], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw2, pl.tile.transpose_view(pl.load(wkv, [0, 512], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw2, pl.tile.transpose_view(pl.load(wkv, [512, 512], [512, 256], target_memory=pl.Mem.Mat)))
        pxw3 = pl.load(x, [publish_base, 768], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw3, pl.tile.transpose_view(pl.load(wkv, [0, 768], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw3, pl.tile.transpose_view(pl.load(wkv, [512, 768], [512, 256], target_memory=pl.Mem.Mat)))
        pxw4 = pl.load(x, [publish_base, 1024], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw4, pl.tile.transpose_view(pl.load(wkv, [0, 1024], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw4, pl.tile.transpose_view(pl.load(wkv, [512, 1024], [512, 256], target_memory=pl.Mem.Mat)))
        pxw5 = pl.load(x, [publish_base, 1280], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw5, pl.tile.transpose_view(pl.load(wkv, [0, 1280], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw5, pl.tile.transpose_view(pl.load(wkv, [512, 1280], [512, 256], target_memory=pl.Mem.Mat)))
        pxw6 = pl.load(x, [publish_base, 1536], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw6, pl.tile.transpose_view(pl.load(wkv, [0, 1536], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw6, pl.tile.transpose_view(pl.load(wkv, [512, 1536], [512, 256], target_memory=pl.Mem.Mat)))
        pxw7 = pl.load(x, [publish_base, 1792], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw7, pl.tile.transpose_view(pl.load(wkv, [0, 1792], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw7, pl.tile.transpose_view(pl.load(wkv, [512, 1792], [512, 256], target_memory=pl.Mem.Mat)))
        pxw8 = pl.load(x, [publish_base, 2048], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw8, pl.tile.transpose_view(pl.load(wkv, [0, 2048], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw8, pl.tile.transpose_view(pl.load(wkv, [512, 2048], [512, 256], target_memory=pl.Mem.Mat)))
        pxw9 = pl.load(x, [publish_base, 2304], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw9, pl.tile.transpose_view(pl.load(wkv, [0, 2304], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw9, pl.tile.transpose_view(pl.load(wkv, [512, 2304], [512, 256], target_memory=pl.Mem.Mat)))
        pxw10 = pl.load(x, [publish_base, 2560], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw10, pl.tile.transpose_view(pl.load(wkv, [0, 2560], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw10, pl.tile.transpose_view(pl.load(wkv, [512, 2560], [512, 256], target_memory=pl.Mem.Mat)))
        pxw11 = pl.load(x, [publish_base, 2816], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw11, pl.tile.transpose_view(pl.load(wkv, [0, 2816], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw11, pl.tile.transpose_view(pl.load(wkv, [512, 2816], [512, 256], target_memory=pl.Mem.Mat)))
        pxw12 = pl.load(x, [publish_base, 3072], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw12, pl.tile.transpose_view(pl.load(wkv, [0, 3072], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw12, pl.tile.transpose_view(pl.load(wkv, [512, 3072], [512, 256], target_memory=pl.Mem.Mat)))
        pxw13 = pl.load(x, [publish_base, 3328], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw13, pl.tile.transpose_view(pl.load(wkv, [0, 3328], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw13, pl.tile.transpose_view(pl.load(wkv, [512, 3328], [512, 256], target_memory=pl.Mem.Mat)))
        pxw14 = pl.load(x, [publish_base, 3584], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw14, pl.tile.transpose_view(pl.load(wkv, [0, 3584], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw14, pl.tile.transpose_view(pl.load(wkv, [512, 3584], [512, 256], target_memory=pl.Mem.Mat)))
        pxw15 = pl.load(x, [publish_base, 3840], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw15, pl.tile.transpose_view(pl.load(wkv, [0, 3840], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw15, pl.tile.transpose_view(pl.load(wkv, [512, 3840], [512, 256], target_memory=pl.Mem.Mat)))
        pl.store(pk0, [0, 0], pub_kv)
        pl.store(pk1, [0, 512], pub_kv)
        pk0 = pl.create_tile([16, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        pk1 = pl.create_tile([16, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        pxw0 = pl.load(x, [publish_base, 0], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul(pxw0, pl.tile.transpose_view(pl.load(wgate, [0, 0], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul(pxw0, pl.tile.transpose_view(pl.load(wgate, [512, 0], [512, 256], target_memory=pl.Mem.Mat)))
        pxw1 = pl.load(x, [publish_base, 256], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw1, pl.tile.transpose_view(pl.load(wgate, [0, 256], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw1, pl.tile.transpose_view(pl.load(wgate, [512, 256], [512, 256], target_memory=pl.Mem.Mat)))
        pxw2 = pl.load(x, [publish_base, 512], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw2, pl.tile.transpose_view(pl.load(wgate, [0, 512], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw2, pl.tile.transpose_view(pl.load(wgate, [512, 512], [512, 256], target_memory=pl.Mem.Mat)))
        pxw3 = pl.load(x, [publish_base, 768], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw3, pl.tile.transpose_view(pl.load(wgate, [0, 768], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw3, pl.tile.transpose_view(pl.load(wgate, [512, 768], [512, 256], target_memory=pl.Mem.Mat)))
        pxw4 = pl.load(x, [publish_base, 1024], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw4, pl.tile.transpose_view(pl.load(wgate, [0, 1024], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw4, pl.tile.transpose_view(pl.load(wgate, [512, 1024], [512, 256], target_memory=pl.Mem.Mat)))
        pxw5 = pl.load(x, [publish_base, 1280], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw5, pl.tile.transpose_view(pl.load(wgate, [0, 1280], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw5, pl.tile.transpose_view(pl.load(wgate, [512, 1280], [512, 256], target_memory=pl.Mem.Mat)))
        pxw6 = pl.load(x, [publish_base, 1536], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw6, pl.tile.transpose_view(pl.load(wgate, [0, 1536], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw6, pl.tile.transpose_view(pl.load(wgate, [512, 1536], [512, 256], target_memory=pl.Mem.Mat)))
        pxw7 = pl.load(x, [publish_base, 1792], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw7, pl.tile.transpose_view(pl.load(wgate, [0, 1792], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw7, pl.tile.transpose_view(pl.load(wgate, [512, 1792], [512, 256], target_memory=pl.Mem.Mat)))
        pxw8 = pl.load(x, [publish_base, 2048], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw8, pl.tile.transpose_view(pl.load(wgate, [0, 2048], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw8, pl.tile.transpose_view(pl.load(wgate, [512, 2048], [512, 256], target_memory=pl.Mem.Mat)))
        pxw9 = pl.load(x, [publish_base, 2304], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw9, pl.tile.transpose_view(pl.load(wgate, [0, 2304], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw9, pl.tile.transpose_view(pl.load(wgate, [512, 2304], [512, 256], target_memory=pl.Mem.Mat)))
        pxw10 = pl.load(x, [publish_base, 2560], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw10, pl.tile.transpose_view(pl.load(wgate, [0, 2560], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw10, pl.tile.transpose_view(pl.load(wgate, [512, 2560], [512, 256], target_memory=pl.Mem.Mat)))
        pxw11 = pl.load(x, [publish_base, 2816], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw11, pl.tile.transpose_view(pl.load(wgate, [0, 2816], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw11, pl.tile.transpose_view(pl.load(wgate, [512, 2816], [512, 256], target_memory=pl.Mem.Mat)))
        pxw12 = pl.load(x, [publish_base, 3072], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw12, pl.tile.transpose_view(pl.load(wgate, [0, 3072], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw12, pl.tile.transpose_view(pl.load(wgate, [512, 3072], [512, 256], target_memory=pl.Mem.Mat)))
        pxw13 = pl.load(x, [publish_base, 3328], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw13, pl.tile.transpose_view(pl.load(wgate, [0, 3328], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw13, pl.tile.transpose_view(pl.load(wgate, [512, 3328], [512, 256], target_memory=pl.Mem.Mat)))
        pxw14 = pl.load(x, [publish_base, 3584], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw14, pl.tile.transpose_view(pl.load(wgate, [0, 3584], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw14, pl.tile.transpose_view(pl.load(wgate, [512, 3584], [512, 256], target_memory=pl.Mem.Mat)))
        pxw15 = pl.load(x, [publish_base, 3840], [16, 256], target_memory=pl.Mem.Mat, clamp=True)
        pk0 = pl.matmul_acc(pk0, pxw15, pl.tile.transpose_view(pl.load(wgate, [0, 3840], [512, 256], target_memory=pl.Mem.Mat)))
        pk1 = pl.matmul_acc(pk1, pxw15, pl.tile.transpose_view(pl.load(wgate, [512, 3840], [512, 256], target_memory=pl.Mem.Mat)))
        P0 = pl.load(ape, [(pl.read(position_ids, [publish_base + 0]) % COMPRESS_RATIO), 0], [1, 512])
        P1 = pl.load(ape, [(pl.read(position_ids, [publish_base + 1]) % COMPRESS_RATIO), 0], [1, 512])
        P2 = pl.load(ape, [(pl.read(position_ids, [publish_base + 2]) % COMPRESS_RATIO), 0], [1, 512])
        P3 = pl.load(ape, [(pl.read(position_ids, [publish_base + 3]) % COMPRESS_RATIO), 0], [1, 512])
        p4 = pl.concat(pl.concat(pl.concat(P0, P1), P2), P3)
        apePW0 = pl.reshape(pl.concat(pl.concat(pl.concat(p4, p4), p4), p4), [16, 512])
        Q0 = pl.load(ape, [(pl.read(position_ids, [publish_base + 0]) % COMPRESS_RATIO), 512], [1, 512])
        Q1 = pl.load(ape, [(pl.read(position_ids, [publish_base + 1]) % COMPRESS_RATIO), 512], [1, 512])
        Q2 = pl.load(ape, [(pl.read(position_ids, [publish_base + 2]) % COMPRESS_RATIO), 512], [1, 512])
        Q3 = pl.load(ape, [(pl.read(position_ids, [publish_base + 3]) % COMPRESS_RATIO), 512], [1, 512])
        q4 = pl.concat(pl.concat(pl.concat(Q0, Q1), Q2), Q3)
        apePW1 = pl.reshape(pl.concat(pl.concat(pl.concat(q4, q4), q4), q4), [16, 512])
        pl.store(pl.add(pk0, apePW0), [0, 0], pub_sc)
        pl.store(pl.add(pk1, apePW1), [0, 512], pub_sc)
        for jo in pl.unroll(STATE_LEN):
            j = publish_base + jo
            pdst = pl.read(state_slot_mapping, [j])
            if pdst >= 0:
                pub_blk = pdst // CSA_STATE_BLOCK_SIZE
                pub_intra = pdst % CSA_STATE_BLOCK_SIZE
                kv_t = pl.load(pub_kv, [jo, 0], [1, OUT_DIM])
                sc_t = pl.load(pub_sc, [jo, 0], [1, OUT_DIM])
                pl.store(pl.reshape(kv_t, [1, 1, OUT_DIM]), [pub_blk, pub_intra, 0], compress_state)
                pl.store(pl.reshape(sc_t, [1, 1, OUT_DIM]), [pub_blk, pub_intra, OUT_DIM], compress_state)

@pl.jit(auto_scope=False)
def _compressor_ratio4(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    """Compress packed requests independently through ordered 512-row state tiles."""
    x.bind_dynamic(0, T_DYN)
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    compress_state.bind_dynamic(0, STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    rope_dup_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_swap_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_sign_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_even_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.INT32)
    rope_odd_idx_template = pl.create_tensor([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_rope_index_prepare"):
        for rope_r in pl.range(PACKED_RMS_TILE):
            for rope_c in pl.range(ROPE_HEAD_DIM):
                rope_lane = rope_c % 2
                pl.write(
                    rope_dup_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c // 2, pl.INT32),
                )
                pl.write(
                    rope_swap_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c + 1 - rope_lane * 2, pl.INT32),
                )
                pl.write(
                    rope_sign_template,
                    [rope_r, rope_c],
                    pl.cast(pl.cast(rope_lane * 2 - 1, pl.INT32), pl.FP32),
                )
        for rope_r2 in pl.range(PACKED_RMS_TILE):
            for rope_c2 in pl.range(ROPE_HEAD_DIM // 2):
                pl.write(rope_even_idx_template, [rope_r2, rope_c2], pl.cast(2 * rope_c2, pl.INT32))
                pl.write(rope_odd_idx_template, [rope_r2, rope_c2], pl.cast(2 * rope_c2 + 1, pl.INT32))

    state_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_state_order_init"):
        pl.write(state_order_fence, [0], pl.cast(0, pl.INT32))
    for request in pl.range(request_count):
        request_start = pl.cast(pl.read(query_start_loc, [request]), pl.INDEX)
        request_end = pl.cast(pl.read(query_start_loc, [request + 1]), pl.INDEX)
        request_table = compress_state_block_table[request]
        for request_offset in pl.range(0, request_end - request_start, PREFILL_STATE_TILE):
            tile_base = request_start + request_offset
            tile_rows = pl.min(PREFILL_STATE_TILE, request_end - tile_base)
            with pl.scope():
                _prefill_compressor_ratio4_tile(
                    x,
                    compress_state,
                    request_table,
                    wkv,
                    wgate,
                    ape,
                    norm_w,
                    cmp_freqs_cos,
                    cmp_freqs_sin,
                    cmp_kv,
                    position_ids,
                    cmp_slot_mapping,
                    state_slot_mapping,
                    rope_dup_idx_template,
                    rope_swap_idx_template,
                    rope_sign_template,
                    rope_even_idx_template,
                    rope_odd_idx_template,
                    state_order_fence,
                    tile_base,
                    tile_rows,
                )
    return cmp_kv, compress_state


prefill_compressor_ratio4_test = _compressor_ratio4


# fixture helpers (copied from models/deepseek_v4_flash_dspark/utils.py)
def _torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported RoPE table dtype: {dtype!r}")


def rope_profile_for_compress_ratio(config: Any, compress_ratio: int) -> tuple[float, int]:
    """Return ``(base_theta, original_seq_len)`` for the two DeepSeek-V4 RoPE profiles."""
    if compress_ratio:
        return float(config.compress_rope_theta), int(config.original_max_position_embeddings)
    return float(config.rope_theta), 0


def _linear_ramp_factor(low: int, high: int, dim: int, *, device: torch.device | None = None) -> torch.Tensor:
    if low == high:
        high = high + 0.001
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


def _find_correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_seq_len: int,
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def token_local_rope(
    config: Any,
    compress_ratio: int,
    position_ids: torch.Tensor,
    *,
    max_seq_len: int | None = None,
    rope_dim: int | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only the RoPE rows used by the supplied absolute positions."""
    dim = int(rope_dim if rope_dim is not None else config.qk_rope_head_dim)
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    positions_i64 = position_ids.to(torch.int64).reshape(-1)
    seq_len = int(
        max_seq_len if max_seq_len is not None else config.max_position_embeddings
    )
    if bool((positions_i64 < 0).any()) or bool((positions_i64 >= seq_len).any()):
        raise ValueError(f"RoPE positions must be in [0, {seq_len})")

    out_dtype = _torch_dtype(dtype)
    out_device = torch.device(device) if device is not None else position_ids.device
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    half_dim = dim // 2
    inv_freq = 1.0 / (
        float(base)
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=out_device) / dim)
    )
    if original_seq_len > 0:
        low, high = _find_correction_range(
            int(config.beta_fast),
            int(config.beta_slow),
            dim,
            float(base),
            int(original_seq_len),
        )
        smooth = 1 - _linear_ramp_factor(low, high, half_dim, device=out_device)
        inv_freq = (
            inv_freq / float(config.rope_factor) * (1 - smooth)
            + inv_freq * smooth
        )
    positions = positions_i64.to(device=out_device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    return (
        torch.cat([cos_half, cos_half], dim=-1).to(out_dtype).contiguous(),
        torch.cat([sin_half, sin_half], dim=-1).to(out_dtype).contiguous(),
    )


def golden_prefill_compressor_ratio4(tensors):
    """Physical token-major torch reference with transactional state publication."""
    token_count = tensors["x"].shape[0]
    x = tensors["x"].view(token_count, D).float()
    compress_state_flat = tensors["compress_state"].view(-1, COMPRESS_STATE_DIM)
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["compress_state_block_table"][0]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cmp_kv = tensors["cmp_kv"]
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    position_ids = tensors["position_ids"]

    kv_proj = x @ wkv.t()  # wkv stored [OUT_DIM, D] for b_trans
    score_proj = x @ wgate.t()

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * CSA_STATE_BLOCK_SIZE + intra

    for tile_base in range(0, token_count, PREFILL_STATE_TILE):
        tile_end = min(tile_base + PREFILL_STATE_TILE, token_count)

        for token_id in range(tile_base, tile_end):
            dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
            if dst_row < 0:
                continue
            write_pos = int(position_ids[token_id].item())
            cur_start = write_pos + 1 - COMPRESS_RATIO
            prev_start = cur_start - COMPRESS_RATIO
            pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
            pool_score = torch.full(
                (STATE_LEN, HEAD_DIM),
                float("-inf"),
                dtype=torch.float32,
            )

            for s in range(COMPRESS_RATIO):
                if write_pos >= STATE_LEN - 1:
                    prev_row = state_row(prev_start + s)
                    if prev_row >= 0:
                        pool_kv[s] = kv_state_flat[prev_row, :HEAD_DIM]
                        pool_score[s] = score_state_flat[prev_row, :HEAD_DIM]

                cur_row = state_row(cur_start + s)
                if cur_row >= 0:
                    pool_kv[COMPRESS_RATIO + s] = kv_state_flat[
                        cur_row,
                        HEAD_DIM:OUT_DIM,
                    ]
                    pool_score[COMPRESS_RATIO + s] = score_state_flat[
                        cur_row,
                        HEAD_DIM:OUT_DIM,
                    ]

            for pool_token_id in range(tile_base, tile_end):
                if int(tensors["state_slot_mapping"][pool_token_id].item()) < 0:
                    continue
                pool_pos = int(position_ids[pool_token_id].item())
                if pool_pos < prev_start or pool_pos > write_pos:
                    continue
                ape_slot = pool_pos % COMPRESS_RATIO
                if pool_pos < cur_start:
                    pool_slot = pool_pos - prev_start
                    pool_kv[pool_slot] = kv_proj[pool_token_id, :HEAD_DIM]
                    pool_score[pool_slot] = score_proj[pool_token_id, :HEAD_DIM] + ape[
                        ape_slot,
                        :HEAD_DIM,
                    ]
                else:
                    pool_slot = COMPRESS_RATIO + pool_pos - cur_start
                    pool_kv[pool_slot] = kv_proj[pool_token_id, HEAD_DIM:OUT_DIM]
                    pool_score[pool_slot] = score_proj[pool_token_id, HEAD_DIM:OUT_DIM] + ape[
                        ape_slot,
                        HEAD_DIM:OUT_DIM,
                    ]

            init_slot = STATE_LEN - 1
            mi = pool_score[init_slot : init_slot + 1].clone()
            li = torch.exp(mi - mi)
            oi = pool_kv[init_slot : init_slot + 1].clone()
            for slot_i in range(STATE_LEN - 1):
                if slot_i < COMPRESS_RATIO and write_pos < STATE_LEN - 1:
                    continue
                slot_score = pool_score[slot_i : slot_i + 1]
                slot_kv = pool_kv[slot_i : slot_i + 1]
                mi_next = torch.maximum(mi, slot_score)
                alpha = torch.exp(mi - mi_next)
                beta = torch.exp(slot_score - mi_next)
                li = alpha * li + beta
                oi = oi * alpha + slot_kv * beta
                mi = mi_next
            pooled = oi / li
            inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
            normed = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
            rope_pair = normed[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
            rope_even = rope_pair[..., 0]
            rope_odd = rope_pair[..., 1]
            cos = tensors["cmp_freqs_cos"][token_id : token_id + 1, 0 : ROPE_HEAD_DIM // 2].float()
            sin = tensors["cmp_freqs_sin"][token_id : token_id + 1, 0 : ROPE_HEAD_DIM // 2].float()
            rot_even = rope_even * cos - rope_odd * sin
            rot_odd = rope_even * sin + rope_odd * cos
            normed[:, NOPE_HEAD_DIM:HEAD_DIM] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
            cache_rows[dst_row] = normed.to(torch.bfloat16)[0]

        publish_base = max(tile_base, tile_end - STATE_LEN)
        for token_id in range(publish_base, tile_end):
            dst_row = int(tensors["state_slot_mapping"][token_id].item())
            if dst_row < 0:
                continue
            pos = int(position_ids[token_id].item())
            kv_state_flat[dst_row] = kv_proj[token_id]
            score_state_flat[dst_row] = score_proj[token_id] + ape[pos % COMPRESS_RATIO]


def build_tensor_specs(start_pos: int = START_POS, token_count: int = PREFILL_SEQ):
    if token_count <= 0 or token_count > MAX_SEQ_LEN:
        raise ValueError(f"token_count must be in [1, {MAX_SEQ_LEN}], got {token_count}")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + token_count > MAX_SEQ_LEN:
        raise ValueError("start_pos + token_count exceeds max_position_embeddings")

    def init_compress_state_block_table():
        logical_blocks = torch.arange(CSA_STATE_MAX_BLOCKS, dtype=torch.int64)
        physical_blocks = (logical_blocks * 17 + 3) % CSA_STATE_BLOCKS_PER_REQUEST
        physical_blocks = physical_blocks.to(torch.int32)
        return physical_blocks.unsqueeze(0)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        physical_block = (block * 17 + 3) % CSA_STATE_BLOCKS_PER_REQUEST
        return physical_block * CSA_STATE_BLOCK_SIZE + intra

    def init_x():
        return ((torch.rand(token_count, D) - 0.5) * 0.1).to(torch.bfloat16)

    def init_compress_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        flat = state.view(-1, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state

    # BF16 weight std and RMSNorm gamma mean/std, averaged over DeepSeek-V4-Flash-0731
    # layers 8/32 (the ratio-4 CSA main compressor). Mirrors decode_compressor_ratio4.
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0240

    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0381

    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1226

    def init_norm_w():
        return 0.9569 + 0.1916 * torch.randn(HEAD_DIM)

    def init_cmp_kv():
        return torch.zeros(CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)

    def init_position_ids():
        return torch.arange(start_pos, start_pos + token_count, dtype=torch.int32)

    def init_cmp_rope_positions():
        positions = init_position_ids().to(torch.int64)
        boundary = (positions + 1) % COMPRESS_RATIO == 0
        return torch.where(boundary, positions - (COMPRESS_RATIO - 1), torch.zeros_like(positions))

    def init_cmp_freqs_cos():
        cos, _ = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return cos.contiguous()

    def init_cmp_freqs_sin():
        _, sin = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return sin.contiguous()

    def init_cmp_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for t in range(token_count):
            pos = start_pos + t
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = (pos + 1) // COMPRESS_RATIO - 1
                if dst_row >= CMP_MAX_BLOCKS * BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone cmp_kv capacity")
                mapping[t] = dst_row
        return mapping

    def init_state_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for t in range(token_count):
            mapping[t] = state_row(start_pos + t)
        return mapping

    return [
        TensorSpec("x", [token_count, D], torch.bfloat16, init_value=init_x),
        TensorSpec("query_start_loc", [2], torch.int32, init_value=torch.tensor([0, token_count], dtype=torch.int32)),
        TensorSpec(
            "compress_state",
            [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [1, CSA_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cmp_freqs_cos", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_cos),
        TensorSpec("cmp_freqs_sin", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_sin),
        TensorSpec(
            "cmp_kv",
            [CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("position_ids", [token_count], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [token_count], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [token_count], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone physical-dynamic DeepSeek V4 prefill compressor ratio4 validation."
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only; also the implicit behavior on the *sim platforms CI uses.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help="Fixture-only absolute position for token 0; not a JIT kernel parameter.",
    )
    parser.add_argument(
        "--token-count",
        "--num-tokens",
        dest="token_count",
        type=int,
        default=PREFILL_SEQ,
        help="Physical token rows, up to 8192.",
    )
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="Persist inputs and golden outputs under the run's data dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for build artifacts; the data dir is <out-dir>/data.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    # Resolve user paths before anchoring the cwd; harness artifacts then land
    # in the repository's gitignored build_output/ regardless of launch cwd.
    golden_data = os.path.abspath(args.golden_data) if args.golden_data else None
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    os.chdir(REPO_ROOT)

    result = run(
        fn=prefill_compressor_ratio4_test,
        specs=build_tensor_specs(args.start_pos, args.token_count),
        golden_fn=golden_prefill_compressor_ratio4,
        golden_data=golden_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            save_kernels=out_dir is not None,
            save_kernels_dir=out_dir,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        save_data=args.save_data,
        compile_only=args.compile_only,
        compare_fn={
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
