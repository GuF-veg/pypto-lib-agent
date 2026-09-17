# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Data movement - pl.load, pl.store, tensor slice read/write, pl.assemble, pl.create_tensor.

Verified entries (all PASS on a2a3, device 0):
  copy_slice        slice read x[r:r+TILE, :] + slice write y[r:r+TILE, :] = ...  (Tensor assemble)
  offset_and_slice  pl.load with explicit offsets + pl.store  vs  slice read + slice write
  col_block         column-block access: x[:, c0:c0+TILE]  vs  pl.load(x, [0, c0], [ROWS, TILE])
  store_ignored     pl.store called as a bare statement (return value ignored)
  store_rebind      y = pl.store(tile, [r, 0], y) - the returned destination reused
  store_dead_bind   written = pl.store(...) with the result unused
  assemble_region   pl.assemble(y_tensor, x_subview, [r, 0]) - dst-first, Tensor-to-Tensor
  acc_tensor        pl.create_tensor accumulator: pl.matmul_acc split-K + pl.add, written out
                    by slice assignment

Rejected forms are collected in PROBES so their exact error text is reproducible with --probe 1.

Tiling: ROWS=128, COLS=128, TILE=64.
"""
import pypto.language as pl

ROWS = 128
COLS = 128
TILE = 64


@pl.jit
def copy_slice(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy"):
            tile = x[r : r + TILE, :]
            y[r : r + TILE, :] = tile
    return y


def golden_copy_slice(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def offset_and_slice(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_off: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_slice: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="offset_load"):
            tile = pl.load(x, [r, 0], [TILE, COLS])
            pl.store(tile, [r, 0], y_off)
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="slice_load"):
            tile = x[r : r + TILE, :]
            y_slice[r : r + TILE, :] = tile
    return y_off, y_slice


def golden_offset_and_slice(tensors):
    tensors["y_off"][:] = tensors["x"][:]
    tensors["y_slice"][:] = tensors["x"][:]


@pl.jit
def col_block(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="col_slice"):
        left = x[:, 0:TILE]
        y[:, 0:TILE] = left
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="col_offset"):
        right = pl.load(x, [0, TILE], [ROWS, TILE])
        pl.store(right, [0, TILE], y)
    return y


def golden_col_block(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def store_ignored(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_ignored"):
            tile = pl.load(x, [r, 0], [TILE, COLS])
            pl.store(tile, [r, 0], y)
    return y


def golden_store_ignored(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def store_rebind(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_rebind"):
            tile = pl.load(x, [r, 0], [TILE, COLS])
            y = pl.store(tile, [r, 0], y)
    return y


def golden_store_rebind(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def store_dead_bind(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_dead_bind"):
            tile = pl.load(x, [r, 0], [TILE, COLS])
            pl.store(tile, [r, 0], y)
    return y


def golden_store_dead_bind(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def assemble_region(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="assemble"):
            src = x[r : r + TILE, :]
            pl.assemble(y, src, [r, 0])
    return y


def golden_assemble_region(tensors):
    tensors["y"][:] = tensors["x"][:]


@pl.jit
def acc_tensor(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        for c in pl.range(0, COLS, TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="acc_tensor"):
                acc = pl.create_tensor([TILE, TILE], dtype=pl.FP32)
                for k in pl.range(0, COLS, TILE):
                    lhs = x[r : r + TILE, k : k + TILE]
                    rhs = x[k : k + TILE, c : c + TILE]
                    acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k == 0))
                bias = x[r : r + TILE, c : c + TILE]
                acc = pl.add(acc, bias)
                y[r : r + TILE, c : c + TILE] = acc
    return y


def golden_acc_tensor(tensors):
    tensors["y"][:] = tensors["x"] @ tensors["x"] + tensors["x"]


_XY = [("x", [ROWS, COLS], "float32", "randn"), ("y", [ROWS, COLS], "float32", None)]

# (name, fn, spec_rows, golden_fn) - every one of these must PASS.
ENTRIES = [
    ("copy_slice", copy_slice, _XY, golden_copy_slice),
    ("offset_and_slice", offset_and_slice,
     [("x", [ROWS, COLS], "float32", "randn"), ("y_off", [ROWS, COLS], "float32", None),
      ("y_slice", [ROWS, COLS], "float32", None)],
     golden_offset_and_slice),
    ("col_block", col_block, _XY, golden_col_block),
    ("store_ignored", store_ignored, _XY, golden_store_ignored),
    ("store_rebind", store_rebind, _XY, golden_store_rebind),
    ("store_dead_bind", store_dead_bind, _XY, golden_store_dead_bind),
    ("assemble_region", assemble_region, _XY, golden_assemble_region),
    ("acc_tensor", acc_tensor, _XY, golden_acc_tensor),
]


@pl.jit
def _probe_slice_write_tile(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """PROBE: slice-assign a pl.load Tile."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="p1"):
            tile = pl.load(x, [r, 0], [TILE, COLS])
            y[r : r + TILE, :] = tile
    return y


@pl.jit
def _probe_assemble_tile_source(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """PROBE: pl.assemble with a Tile source into a Tensor target."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="p2"):
            tile = pl.load(x, [r, 0], [TILE, COLS])
            pl.assemble(y, tile, [r, 0])
    return y


@pl.jit
def _probe_assemble_src_first(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """PROBE: pl.assemble with the arguments swapped (source first)."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="p3"):
            src = x[r : r + TILE, :]
            pl.assemble(src, y, [r, 0])
    return y


@pl.jit
def _probe_create_tensor_mix_tiles(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """PROBE: pl.create_tensor accumulator mixed with pl.load Tile operands."""
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="p4"):
            lhs = pl.load(x, [r, 0], [TILE, COLS])
            rhs = pl.load(x, [0, 0], [COLS, TILE])
            acc = pl.create_tensor([TILE, TILE], dtype=pl.FP32)
            acc = pl.matmul_acc(acc, lhs, rhs, init_cond=True)
            pl.store(acc, [r, 0], y)
    return y


@pl.jit
def _probe_create_tensor_store(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """PROBE: pl.store with a pl.create_tensor result as the source."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="p5"):
        acc = pl.create_tensor([TILE, TILE], dtype=pl.FP32)
        pl.store(acc, [0, 0], y)
    return y


@pl.jit
def _probe_create_tensor_init_value(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """PROBE: pl.create_tensor with init_value (removed from the runtime)."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="p6"):
        acc = pl.create_tensor([TILE, TILE], dtype=pl.FP32, init_value=0.0)
        pl.store(acc, [0, 0], y)
    return y


@pl.jit
def _probe_acc_seeded_by_load(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """PROBE: use a loaded Tile as the accumulator of pl.matmul_acc."""
    for r in pl.parallel(0, ROWS, TILE):
        for c in pl.range(0, COLS, TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="p7"):
                acc = pl.load(x, [r, c], [TILE, TILE])
                for k in pl.range(0, COLS, TILE):
                    lhs = pl.load(x, [r, k], [TILE, TILE])
                    rhs = pl.load(x, [k, c], [TILE, TILE])
                    acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k == 0))
                pl.store(acc, [r, c], y)
    return y


# (name, fn, golden_fn) - expected to be REJECTED; the error text is the finding.
PROBES = [
    ("probe_slice_write_tile", _probe_slice_write_tile, golden_copy_slice),
    ("probe_assemble_tile_source", _probe_assemble_tile_source, golden_copy_slice),
    ("probe_assemble_src_first", _probe_assemble_src_first, golden_copy_slice),
    ("probe_create_tensor_mix_tiles", _probe_create_tensor_mix_tiles, golden_copy_slice),
    ("probe_create_tensor_store", _probe_create_tensor_store, golden_copy_slice),
    ("probe_create_tensor_init_value", _probe_create_tensor_init_value, golden_copy_slice),
    ("probe_acc_seeded_by_load", _probe_acc_seeded_by_load, golden_copy_slice),
]


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
    for name, fn, spec_rows, golden_fn in ENTRIES:
        print(f"[ENTRY] {name}")
        try:
            result = run(fn=fn, specs=build_specs(spec_rows), golden_fn=golden_fn,
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
        for name, fn, golden_fn in PROBES:
            print(f"[PROBE] {name}")
            try:
                result = run(fn=fn, specs=build_specs(_XY), golden_fn=golden_fn,
                             config=config, rtol=1e-5, atol=1e-5)
                print(f"[PROBE] {name} RESULT passed={result.passed} error={result.error}")
            except Exception as exc:  # noqa: BLE001 - the error text is the finding
                print(f"[PROBE] {name} REJECTED {type(exc).__name__}: {exc}")

    if failed:
        print(f"[ENTRY] FAILED: {failed}")
        raise SystemExit(1)
    print(f"[ENTRY] ALL PASS: {[e[0] for e in ENTRIES]}")