# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor-level operators and scheduling hints.

  full_const          pl.full(shape, dtype=, value=) materializes a constant tensor
  arange_row          pl.arange(start, [1, N], dtype=INT32) - one row of 0..N-1
  dim_loop            pl.tensor.dim(x, 0) as a runtime loop bound for pl.range
  tensor_scalar_read  pl.read(tensor, [i, j]) inside pl.at, broadcast with pl.add
  incore_tile_rw      @pl.jit.incore + pl.tile.get_block_idx + pl.read/pl.write on a Tile,
                      dispatched with `with pl.spmd(n)`
  incore_idx_alias    pl.get_block_idx() (tensor-scope alias) inside @pl.jit.incore
  spmd_loop_idx       `for i in pl.spmd(n)` - `i` and pl.get_block_idx() both index the grid
  block_num_addr      pl.get_block_num() used in addressing (must equal the grid size)
  block_num_value     pl.get_block_num() written out to an INT32 tensor
  create_manual_dep   pl.create_tensor(..., manual_dep=True) scratch + explicit deps
  hint_dump_tag       pl.dump_tag(x) statement before a dispatch
  hint_cache_default  pl.set_cache_policy(x, pl.CachePolicy.DEFAULT) in a pl.at scope

Rejected create_tensor keywords, pl.no_dep call forms and the BYPASS cache policy
are collected in PROBES and printed with --probe 1.

Tiling: ROWS=128, COLS=128, TILE=32 (BLOCKS=4).
"""
import pypto.language as pl

ROWS = 128
COLS = 128
TILE = 32
BLOCKS = ROWS // TILE


@pl.jit
def full_const(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y2: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    out_i32: pl.Out[pl.Tensor[[1, COLS], pl.INT32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_const"):
            c = pl.full([TILE, COLS], dtype=pl.FP32, value=1.5)
            d = pl.full([TILE, COLS], dtype=pl.FP32, value=-2.25)
            y[r : r + TILE, :] = pl.add(x[r : r + TILE, :], c)
            y2[r : r + TILE, :] = pl.mul(x[r : r + TILE, :], d)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_const_i32"):
        out_i32[0:1, :] = pl.full([1, COLS], dtype=pl.INT32, value=7)
    return y, y2, out_i32


def golden_full_const(tensors):
    tensors["y"][:] = tensors["x"] + 1.5
    tensors["y2"][:] = tensors["x"] * -2.25
    tensors["out_i32"][:] = 7


@pl.jit
def arange_row(
    out: pl.Out[pl.Tensor[[1, COLS], pl.INT32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="arange_row"):
        out[0:1, :] = pl.arange(0, [1, COLS], dtype=pl.INT32)
    return out


def golden_arange_row(tensors):
    import torch

    tensors["out"][:] = torch.arange(0, COLS, dtype=torch.int32).reshape(1, COLS)


@pl.jit
def dim_loop(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    rows = pl.tensor.dim(x, 0)
    for r in pl.range(0, rows, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dim_loop"):
            y[r : r + TILE, :] = pl.mul(x[r : r + TILE, :], 2.0)
    return y


def golden_dim_loop(tensors):
    tensors["y"][:] = tensors["x"] * 2.0


@pl.jit
def tensor_scalar_read(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scalar_read"):
            s = pl.read(x, [r, 0])
            y[r : r + TILE, :] = pl.add(x[r : r + TILE, :], s)
    return y


def golden_tensor_scalar_read(tensors):
    import torch

    x = tensors["x"]
    # `r` is the row TILE offset, so the scalar read at [r, 0] is broadcast
    # across every row of that tile.
    tile_idx = torch.arange(0, ROWS, TILE).repeat_interleave(TILE)
    tensors["y"][:] = x + x[tile_idx, 0].reshape(ROWS, 1)


@pl.jit.incore
def tile_scalar_rw(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    r0 = pl.tile.get_block_idx() * TILE
    t = pl.load(x, [r0, 0], [TILE, COLS])
    v = pl.read(t, [0, 0])
    pl.write(t, [3, 4], v)
    pl.store(t, [r0, 0], y)
    return y


@pl.jit
def incore_tile_rw(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.spmd(BLOCKS, name_hint="incore_tile_rw"):
        y = tile_scalar_rw(x, y)
    return y


def golden_incore_tile_rw(tensors):
    tensors["y"][:] = tensors["x"][:]
    for k in range(BLOCKS):
        tensors["y"][k * TILE + 3, 4] = tensors["x"][k * TILE, 0]


@pl.jit.incore
def copy_idx_alias(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    r0 = pl.get_block_idx() * TILE
    y[r0 : r0 + TILE, :] = x[r0 : r0 + TILE, :]
    return y


@pl.jit
def incore_idx_alias(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.spmd(BLOCKS, name_hint="incore_idx_alias"):
        y = copy_idx_alias(x, y)
    return y


def golden_incore_idx_alias(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def spmd_loop_idx(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_i: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_bi: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for i in pl.spmd(BLOCKS, name_hint="spmd_loop_idx"):
        r_i = i * TILE
        bi = pl.get_block_idx()
        r_b = bi * TILE
        y_i[r_i : r_i + TILE, :] = pl.add(x[r_i : r_i + TILE, :], pl.cast(i, pl.INT32))
        y_bi[r_b : r_b + TILE, :] = pl.add(x[r_b : r_b + TILE, :], pl.cast(bi, pl.INT32))
    return y_i, y_bi


def golden_spmd_loop_idx(tensors):
    import torch

    offs = torch.arange(BLOCKS, dtype=torch.float32).repeat_interleave(TILE).reshape(ROWS, 1)
    tensors["y_i"][:] = tensors["x"] + offs
    tensors["y_bi"][:] = tensors["x"] + offs


@pl.jit
def block_num_addr(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for i in pl.spmd(BLOCKS, name_hint="block_num_addr"):
        n = pl.get_block_num()
        off = (n - BLOCKS) * TILE
        r0 = i * TILE
        y[off + r0 : off + r0 + TILE, :] = x[r0 : r0 + TILE, :]
    return y


def golden_block_num_addr(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def block_num_value(
    out: pl.Out[pl.Tensor[[1, COLS], pl.FP32]],
):
    for i in pl.spmd(BLOCKS, name_hint="block_num_value"):
        n = pl.get_block_num()
        out[0:1, :] = pl.add(pl.full([1, COLS], dtype=pl.FP32, value=0.0), pl.cast(n, pl.INT32))
    return out


def golden_block_num_value(tensors):
    tensors["out"][:] = float(BLOCKS)


@pl.jit.incore
def scale_two(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    r0 = pl.tile.get_block_idx() * TILE
    y[r0 : r0 + TILE, :] = pl.mul(x[r0 : r0 + TILE, :], 2.0)
    return y


@pl.jit.incore
def add_one(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    r0 = pl.tile.get_block_idx() * TILE
    y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], 1.0)
    return y


@pl.jit
def create_manual_dep(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    scratch = pl.create_tensor([ROWS, COLS], dtype=pl.FP32, manual_dep=True)
    with pl.spmd(BLOCKS, name_hint="seed_scratch") as seed_tid:
        scratch = scale_two(x, scratch)
    with pl.spmd(BLOCKS, name_hint="consume_scratch", deps=[seed_tid]):
        y = add_one(scratch, y)
    return y


def golden_create_manual_dep(tensors):
    tensors["y"][:] = tensors["x"] * 2.0 + 1.0


@pl.jit
def hint_dump_tag(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    pl.dump_tag(x)
    with pl.spmd(BLOCKS, name_hint="hint_dump_tag"):
        y = add_one(x, y)
    return y


def golden_hint_dump_tag(tensors):
    tensors["y"][:] = tensors["x"] + 1.0


@pl.jit
def hint_cache_default(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hint_cache_default"):
            pl.set_cache_policy(x, pl.CachePolicy.DEFAULT)
            y[r : r + TILE, :] = pl.add(x[r : r + TILE, :], 1.0)
    return y


def golden_hint_cache_default(tensors):
    tensors["y"][:] = tensors["x"] + 1.0


ENTRIES = [
    ("full_const", full_const, golden_full_const),
    ("arange_row", arange_row, golden_arange_row),
    ("dim_loop", dim_loop, golden_dim_loop),
    ("tensor_scalar_read", tensor_scalar_read, golden_tensor_scalar_read),
    ("incore_tile_rw", incore_tile_rw, golden_incore_tile_rw),
    ("incore_idx_alias", incore_idx_alias, golden_incore_idx_alias),
    ("spmd_loop_idx", spmd_loop_idx, golden_spmd_loop_idx),
    ("block_num_addr", block_num_addr, golden_block_num_addr),
    ("block_num_value", block_num_value, golden_block_num_value),
    ("create_manual_dep", create_manual_dep, golden_create_manual_dep),
    ("hint_dump_tag", hint_dump_tag, golden_hint_dump_tag),
    ("hint_cache_default", hint_cache_default, golden_hint_cache_default),
]

_XY = [("x", [ROWS, COLS], "float32", "randn"), ("y", [ROWS, COLS], "float32", None)]

_ENTRY_SPECS = {
    "full_const": [("x", [ROWS, COLS], "float32", "randn"),
                   ("y", [ROWS, COLS], "float32", None),
                   ("y2", [ROWS, COLS], "float32", None),
                   ("out_i32", [1, COLS], "int32", None)],
    "arange_row": [("out", [1, COLS], "int32", None)],
    "block_num_value": [("out", [1, COLS], "float32", None)],
    "spmd_loop_idx": [("x", [ROWS, COLS], "float32", "randn"),
                      ("y_i", [ROWS, COLS], "float32", None),
                      ("y_bi", [ROWS, COLS], "float32", None)],
}


def build_specs(spec_rows):
    import torch

    from golden import TensorSpec

    specs = []
    for name, shape, dtype, init in spec_rows:
        kwargs = {}
        if init == "randn":
            kwargs["init_value"] = torch.randn
        specs.append(TensorSpec(name, list(shape), getattr(torch, dtype), **kwargs))
    return specs


def specs_for(name):
    return build_specs(_ENTRY_SPECS.get(name, _XY))


# ---------------------------------------------------------------------------
# Rejected / negative forms. Run with --probe 1.
# ---------------------------------------------------------------------------


@pl.jit
def probe_create_memory_space(
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    acc = pl.create_tensor([TILE, COLS], dtype=pl.FP32, memory_space="vec")
    y[0:TILE, :] = acc
    return y


@pl.jit
def probe_create_valid_shape(
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    acc = pl.create_tensor([TILE, COLS], dtype=pl.FP32, valid_shape=[TILE, COLS])
    y[0:TILE, :] = acc
    return y


@pl.jit
def probe_create_init_value(
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    acc = pl.create_tensor([TILE, COLS], dtype=pl.FP32, init_value=0.0)
    y[0:TILE, :] = acc
    return y


@pl.jit
def probe_manual_dep_undeped_read(
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    acc = pl.create_tensor([TILE, COLS], dtype=pl.FP32, manual_dep=True)
    y[0:TILE, :] = acc
    return y


def golden_probe_zero(tensors):
    tensors["y"][:] = 0.0


@pl.jit
def probe_dim_var_axis(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    ax = pl.tensor.dim(x, 0)
    n = pl.tensor.dim(x, ax)
    for r in pl.range(0, n, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_dim_var"):
            y[r : r + TILE, :] = x[r : r + TILE, :]
    return y


@pl.jit
def probe_arange_lead_dim(
    out: pl.Out[pl.Tensor[[TILE, COLS], pl.INT32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_arange_lead"):
        out[0:TILE, :] = pl.arange(0, [TILE, COLS], dtype=pl.INT32)
    return out


def golden_probe_arange_lead_dim(tensors):
    import torch

    row = torch.arange(0, COLS, dtype=torch.int32).reshape(1, COLS)
    tensors["out"][:] = row.repeat(TILE, 1)


def golden_probe_identity(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def probe_no_dep_in_spmd(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.spmd(BLOCKS, name_hint="probe_no_dep_spmd"):
        y = add_one(pl.no_dep(x), y)
    return y


@pl.jit
def probe_no_dep_on_output(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.spmd(BLOCKS, name_hint="probe_no_dep_out"):
        y = add_one(x, pl.no_dep(y))
    return y


@pl.jit
def probe_no_dep_args_kwarg(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_nodep_at", no_dep_args=[x]):
            y[r : r + TILE, :] = x[r : r + TILE, :]
    return y


@pl.jit
def plain_callee(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="plain_callee"):
            y[r : r + TILE, :] = x[r : r + TILE, :]
    return y


@pl.jit
def probe_no_dep_direct_call(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    y = plain_callee(pl.no_dep(x), y)
    return y


@pl.jit
def probe_dump_tag_wrapper(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_dump_wrap"):
            y[r : r + TILE, :] = pl.dump_tag(x[r : r + TILE, :])
    return y


@pl.jit
def probe_cache_bypass(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="probe_cache_bypass"):
            pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
            y[r : r + TILE, :] = pl.add(x[r : r + TILE, :], 1.0)
    return y


@pl.jit
def probe_idx_add_raw(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for i in pl.spmd(BLOCKS, name_hint="probe_idx_raw"):
        r0 = i * TILE
        y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], i)
    return y


@pl.jit
def probe_idx_add_fp32_cast(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for i in pl.spmd(BLOCKS, name_hint="probe_idx_fp32"):
        r0 = i * TILE
        y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], pl.cast(i, pl.FP32))
    return y


@pl.jit
def probe_idx_add_int32_cast(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for i in pl.spmd(BLOCKS, name_hint="probe_idx_int32"):
        r0 = i * TILE
        y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], pl.cast(i, pl.INT32))
    return y


@pl.jit
def probe_idx_scalar_mul_float(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for i in pl.spmd(BLOCKS, name_hint="probe_idx_mul"):
        r0 = i * TILE
        n = pl.get_block_num()
        y[r0 : r0 + TILE, :] = pl.add(x[r0 : r0 + TILE, :], pl.mul(pl.cast(n, pl.INT32), 1.0))
    return y


@pl.jit
def probe_write_index_scalar(
    out: pl.Out[pl.Tensor[[1, BLOCKS], pl.INT32]],
):
    for i in pl.spmd(BLOCKS, name_hint="probe_write_idx"):
        n = pl.get_block_num()
        pl.write(out, [0, i], n)
    return out


def golden_probe_write_index_scalar(tensors):
    tensors["out"][:] = BLOCKS


@pl.jit
def probe_write_casted_scalar(
    out: pl.Out[pl.Tensor[[1, BLOCKS], pl.INT32]],
):
    for i in pl.spmd(BLOCKS, name_hint="probe_write_cast"):
        n = pl.get_block_num()
        pl.write(out, [0, i], pl.cast(n, pl.INT32))
    return out


def golden_probe_write_casted_scalar(tensors):
    tensors["out"][:] = BLOCKS


PROBES = [
    ("create_tensor(memory_space=...)", probe_create_memory_space,
     [("y", [ROWS, COLS], "float32", None)], golden_probe_zero),
    ("create_tensor(valid_shape=...)", probe_create_valid_shape,
     [("y", [ROWS, COLS], "float32", None)], golden_probe_zero),
    ("create_tensor(init_value=0.0)", probe_create_init_value,
     [("y", [ROWS, COLS], "float32", None)], golden_probe_zero),
    ("create_tensor(manual_dep=True), read with no dep", probe_manual_dep_undeped_read,
     [("y", [ROWS, COLS], "float32", None)], golden_probe_zero),
    ("tensor.dim(x, <runtime axis>)", probe_dim_var_axis,
     _XY, golden_probe_identity),
    ("arange([TILE, COLS]) leading dim > 1", probe_arange_lead_dim,
     [("out", [TILE, COLS], "int32", None)], golden_probe_arange_lead_dim),
    ("pl.no_dep(x) in a pl.spmd dispatch", probe_no_dep_in_spmd,
     _XY, golden_hint_dump_tag),
    ("pl.no_dep(y) on the output slot", probe_no_dep_on_output,
     _XY, golden_hint_dump_tag),
    ("pl.at(..., no_dep_args=[x])", probe_no_dep_args_kwarg,
     _XY, golden_probe_identity),
    ("pl.no_dep(x) at a jit-to-jit direct call", probe_no_dep_direct_call,
     _XY, golden_probe_identity),
    ("pl.dump_tag(t) as an expression wrapper", probe_dump_tag_wrapper,
     _XY, golden_probe_identity),
    ("pl.set_cache_policy(x, CachePolicy.BYPASS)", probe_cache_bypass,
     _XY, golden_hint_cache_default),
    ("pl.add(fp32, block_idx) with no cast", probe_idx_add_raw,
     _XY, golden_hint_cache_default),
    ("pl.add(fp32, pl.cast(block_idx, pl.FP32))", probe_idx_add_fp32_cast,
     _XY, golden_hint_cache_default),
    ("pl.add(fp32, pl.cast(block_idx, pl.INT32))", probe_idx_add_int32_cast,
     _XY, golden_hint_cache_default),
    ("pl.mul(pl.cast(block_num, pl.INT32), 1.0)", probe_idx_scalar_mul_float,
     _XY, golden_hint_cache_default),
    ("pl.write(int32, [0, i], block_num_index_scalar)", probe_write_index_scalar,
     [("out", [1, BLOCKS], "int32", None)], golden_probe_write_index_scalar),
    ("pl.write(int32, [0, i], cast(block_num, INT32)) from 4 blocks", probe_write_casted_scalar,
     [("out", [1, BLOCKS], "int32", None)], golden_probe_write_casted_scalar),
]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--probe", type=int, default=0, choices=range(2),
                        help="1 = also run the rejected-form probes and print their errors")
    args = parser.parse_args()

    config = dict(platform=args.platform, device_id=args.device,
                  enable_chip_swimlane=args.enable_chip_swimlane)

    failed = []
    for name, fn, golden_fn in ENTRIES:
        print(f"[ENTRY] {name}")
        try:
            result = run(fn=fn, specs=specs_for(name), golden_fn=golden_fn,
                         config=config, rtol=1e-5, atol=1e-5)
        except Exception as exc:  # noqa: BLE001 - report the exact compiler error
            print(f"[ENTRY] {name} RAISED {type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        if not result.passed:
            if result.error:
                print(result.error)
            failed.append(name)

    if args.probe:
        for name, fn, spec_rows, golden_fn in PROBES:
            print(f"[PROBE] {name}")
            try:
                result = run(fn=fn, specs=build_specs(spec_rows), golden_fn=golden_fn,
                             config=config, rtol=1e-5, atol=1e-5)
                print(f"[PROBE] {name} RESULT passed={result.passed} error={result.error}")
            except Exception as exc:  # noqa: BLE001 - the error text is the finding
                print(f"[PROBE] {name} REJECTED {type(exc).__name__}: {exc}")

    if failed:
        print(f"[ENTRY] FAILED: {failed}")
        raise SystemExit(1)
    print(f"[ENTRY] ALL PASS: {[e[0] for e in ENTRIES]}")