# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Row and column broadcast-expand - one line per operator, then the tiling used.

    pl.row_expand(tile, vec)          [R,1] -> [R,C]: plain copy of vec, tile supplies the shape
    pl.row_expand_add(tile, vec)      tile + vec            (vec may be a tmp-carrying tile form)
    pl.row_expand_sub(tile, vec)      tile - vec
    pl.row_expand_mul(tile, vec)      tile * vec
    pl.row_expand_div(tile, vec)      tile / vec
    pl.row_expand_max(tile, vec)      maximum(tile, vec)
    pl.row_expand_min(tile, vec)      minimum(tile, vec)
    pl.row_expand_expdif(tile, vec)   exp(tile - vec)
    pl.col_expand(tile, vec)          [1,C] -> [R,C]: plain copy of vec, tile supplies the shape
    pl.col_expand_add(tile, vec)      tile + vec
    pl.col_expand_sub(tile, vec)      tile - vec
    pl.col_expand_mul(tile, vec)      tile * vec
    pl.col_expand_div(tile, vec)      tile / vec
    pl.col_expand_max(tile, vec)      maximum(tile, vec)
    pl.col_expand_min(tile, vec)      minimum(tile, vec)
    pl.col_expand_expdif(tile, vec)   exp(tile - vec)
    pl.expands(target, scalar)        scalar -> target shape; NO a2a3 codegen (unusable)
    pl.expand_clone(src, target)      rank-3 clone-expand, one broadcast dim, into target

The [R, C] tile is always the FIRST argument and the carrier ([R, 1] for the row
family, [1, C] for the column family) is always the SECOND; the order is not
swappable. The carrier must be rank >= 2 and is read with the full row/column
extent, so a [1, C] carrier is rejected by every row_expand_* call and vice
versa. Rows are tiled across core-groups via pl.parallel and each pl.at InCore
scope works on one [ROW_TILE, COLS] block; entry 5 additionally shows the tile
form of row_expand_add, whose optional tmp scratch exists only for Tile inputs.

pl.expands has no a2a3 backend codegen ("No codegen registered for operation:
tile.expands"), so it cannot appear in a passing kernel; see the report.

For sub/div always use the explicit form above. Plain pl.sub(rv, tile) with the
carrier first is accepted but silently computes tile - rv, pl.div(rv, tile) is
rejected outright, and plain pl.mul(tile, [1, C]) compiles yet returns wrong
values from row 1 on (only pl.col_expand_mul broadcasts a column vector).
"""
import pypto.language as pl

ROWS = 128
COLS = 128
TILE = 64               # rows per core-group
PACK = 32 // 4          # FP32 elements in one 32-byte lane block

BATCH = 8               # expand_clone: rank-3 case
PLANE = 64

_ROW_OUT_NAMES = ["o_bare", "o_add", "o_sub", "o_mul", "o_div", "o_max", "o_min", "o_expdif"]


# ---------------------------------------------------------------------------
# 1. Normalisation kernel: min-max rescale down each row, then a per-column
#    affine. row_expand_sub / row_expand_div carry the row statistics,
#    col_expand_mul / col_expand_add apply the per-column weight and bias.
# ---------------------------------------------------------------------------
@pl.jit
def expand_normalize(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    gamma: pl.Tensor[[1, COLS], pl.FP32],
    beta: pl.Tensor[[1, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="expand_normalize"):
            tile = x[r : r + TILE, :]
            row_lo = pl.row_min(tile)                       # [TILE, 1]
            row_hi = pl.row_max(tile)                       # [TILE, 1]
            row_span = pl.sub(row_hi, row_lo)               # [TILE, 1]
            centred = pl.row_expand_sub(tile, row_lo)       # tile - row_lo
            unit = pl.row_expand_div(centred, row_span)     # (tile - lo) / (hi - lo)
            scaled = pl.col_expand_mul(unit, gamma[:, :])   # * gamma[c]
            y[r : r + TILE, :] = pl.col_expand_add(scaled, beta[:, :])
    return y


# ---------------------------------------------------------------------------
# 2. Softmax whose shift-and-exponentiate step is the expdif primitive, the
#    same exp(tile - row_max) used when an online softmax rescores a block.
# ---------------------------------------------------------------------------
@pl.jit
def expand_softmax(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="expand_softmax"):
            tile = x[r : r + TILE, :]
            row_max = pl.row_max(tile)                             # [TILE, 1]
            shifted = pl.row_expand_expdif(tile, row_max)          # exp(tile - row_max)
            denom = pl.row_sum(shifted)                            # [TILE, 1]
            y[r : r + TILE, :] = pl.row_expand_div(shifted, denom)
    return y


# ---------------------------------------------------------------------------
# 3. The eight row operators, each into its own output. o_bare is a copy of
#    rv, not a function of x, which is what separates row_expand from
#    row_expand_add.
# ---------------------------------------------------------------------------
@pl.jit
def row_expand_family(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    rv: pl.Tensor[[ROWS, 1], pl.FP32],
    o_bare: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_add: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_sub: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_mul: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_div: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_max: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_min: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_expdif: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="row_expand_family"):
            tile = x[r : r + TILE, :]
            vec = rv[r : r + TILE, :]               # [TILE, 1] row carrier
            o_bare[r : r + TILE, :] = pl.row_expand(tile, vec)
            o_add[r : r + TILE, :] = pl.row_expand_add(tile, vec)
            o_sub[r : r + TILE, :] = pl.row_expand_sub(tile, vec)
            o_mul[r : r + TILE, :] = pl.row_expand_mul(tile, vec)
            o_div[r : r + TILE, :] = pl.row_expand_div(tile, vec)
            o_max[r : r + TILE, :] = pl.row_expand_max(tile, vec)
            o_min[r : r + TILE, :] = pl.row_expand_min(tile, vec)
            o_expdif[r : r + TILE, :] = pl.row_expand_expdif(tile, vec)
    return o_bare, o_add, o_sub, o_mul, o_div, o_max, o_min, o_expdif


# ---------------------------------------------------------------------------
# 4. The eight column operators, same eight shapes with a [1, C] carrier.
# ---------------------------------------------------------------------------
@pl.jit
def col_expand_family(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    cv: pl.Tensor[[1, COLS], pl.FP32],
    o_bare: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_add: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_sub: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_mul: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_div: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_max: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_min: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    o_expdif: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="col_expand_family"):
            tile = x[r : r + TILE, :]
            vec = cv[:, :]                          # [1, COLS] column carrier
            o_bare[r : r + TILE, :] = pl.col_expand(tile, vec)
            o_add[r : r + TILE, :] = pl.col_expand_add(tile, vec)
            o_sub[r : r + TILE, :] = pl.col_expand_sub(tile, vec)
            o_mul[r : r + TILE, :] = pl.col_expand_mul(tile, vec)
            o_div[r : r + TILE, :] = pl.col_expand_div(tile, vec)
            o_max[r : r + TILE, :] = pl.col_expand_max(tile, vec)
            o_min[r : r + TILE, :] = pl.col_expand_min(tile, vec)
            o_expdif[r : r + TILE, :] = pl.col_expand_expdif(tile, vec)
    return o_bare, o_add, o_sub, o_mul, o_div, o_max, o_min, o_expdif


# ---------------------------------------------------------------------------
# 5. row_expand_add's two carrier forms. The tmp scratch exists only for Tile
#    operands: a Tensor call must omit it ("Tensor inputs must not pass tmp").
#    Without tmp, a [R, 1] carrier broadcasts one scalar per row; a row-major
#    [R, 32/elem_bytes] (here [R, 8]) carrier instead repeats that 32-byte lane
#    block across the destination columns.
# ---------------------------------------------------------------------------
@pl.jit
def row_expand_add_tmp(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    rv: pl.Tensor[[ROWS, 1], pl.FP32],
    rp: pl.Tensor[[ROWS, PACK], pl.FP32],
    y_vec: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_packed: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="row_expand_add_tmp"):
            tile: pl.Tile[[TILE, COLS], pl.FP32] = pl.load(x, [r, 0], [TILE, COLS])
            vec: pl.Tile[[TILE, 1], pl.FP32] = pl.load(rv, [r, 0], [TILE, 1])
            packed: pl.Tile[[TILE, PACK], pl.FP32] = pl.load(rp, [r, 0], [TILE, PACK])
            tmp: pl.Tile[[TILE, COLS], pl.FP32] = pl.tile.create(
                [TILE, COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            out_vec = pl.tile.row_expand_add(tile, vec, tmp=tmp)
            out_packed = pl.tile.row_expand_add(tile, packed)
            y_vec = pl.store(out_vec, [r, 0], y_vec)
            y_packed = pl.store(out_packed, [r, 0], y_packed)
    return y_vec, y_packed


# ---------------------------------------------------------------------------
# 6. expand_clone: a rank-3 tensor op, so it needs an InCore block rather than
#    the row-tiled shape of the other entries, and the target both names the
#    output shape and receives the copy. Exactly one dimension may grow from 1.
# ---------------------------------------------------------------------------
@pl.jit.incore
def _expand_clone_step(
    src: pl.Tensor[[1, PLANE, PLANE], pl.FP32],
    dst: pl.Out[pl.Tensor[[BATCH, PLANE, PLANE], pl.FP32]],
):
    out: pl.Tensor[[BATCH, PLANE, PLANE], pl.FP32] = pl.expand_clone(src, dst)
    return out


@pl.jit
def expand_clone_batch(
    src: pl.Tensor[[1, PLANE, PLANE], pl.FP32],
    dst: pl.Out[pl.Tensor[[BATCH, PLANE, PLANE], pl.FP32]],
):
    dst = _expand_clone_step(src, dst)
    return dst


# ---------------------------------------------------------------------------
# 7. Accepted dtypes for the expand families: FP16, FP32, INT16 and INT32 are
#    executable on a2a3. BF16 is rejected by both the tile IR and PTOAS, and
#    mixed dtypes are rejected even where the type rule would promote.
# ---------------------------------------------------------------------------
@pl.jit
def expand_dtypes(
    xf: pl.Tensor[[ROWS, COLS], pl.FP16],
    vf: pl.Tensor[[ROWS, 1], pl.FP16],
    xi: pl.Tensor[[ROWS, COLS], pl.INT32],
    vi: pl.Tensor[[ROWS, 1], pl.INT32],
    yf: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP16]],
    yi: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="expand_dtypes"):
            yf[r : r + TILE, :] = pl.row_expand_mul(xf[r : r + TILE, :], vf[r : r + TILE, :])
            yi[r : r + TILE, :] = pl.row_expand_add(xi[r : r + TILE, :], vi[r : r + TILE, :])
    return yf, yi


# ---------------------------------------------------------------------------
# Tensor specifications, in each kernel's parameter order.
# ---------------------------------------------------------------------------
def _unit(shape):
    """Values in [0.8, 1.2): keeps div, max/min and exp(tile - vec) bounded."""
    import torch

    return lambda: torch.rand(shape) * 0.4 + 0.8


def spec_expand_normalize():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=_unit([ROWS, COLS])),
        TensorSpec("gamma", [1, COLS], torch.float32, init_value=_unit([1, COLS])),
        TensorSpec("beta", [1, COLS], torch.float32, init_value=_unit([1, COLS])),
        TensorSpec("y", [ROWS, COLS], torch.float32),
    ]


def spec_expand_softmax():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=_unit([ROWS, COLS])),
        TensorSpec("y", [ROWS, COLS], torch.float32),
    ]


def _spec_row_family():
    import torch

    from golden import TensorSpec

    specs = [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=_unit([ROWS, COLS])),
        TensorSpec("rv", [ROWS, 1], torch.float32, init_value=_unit([ROWS, 1])),
    ]
    specs += [TensorSpec(name, [ROWS, COLS], torch.float32) for name in _ROW_OUT_NAMES]
    return specs


def _spec_col_family():
    import torch

    from golden import TensorSpec

    specs = [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=_unit([ROWS, COLS])),
        TensorSpec("cv", [1, COLS], torch.float32, init_value=_unit([1, COLS])),
    ]
    specs += [TensorSpec(name, [ROWS, COLS], torch.float32) for name in _ROW_OUT_NAMES]
    return specs


def spec_row_expand_add_tmp():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=_unit([ROWS, COLS])),
        TensorSpec("rv", [ROWS, 1], torch.float32, init_value=_unit([ROWS, 1])),
        TensorSpec("rp", [ROWS, PACK], torch.float32, init_value=_unit([ROWS, PACK])),
        TensorSpec("y_vec", [ROWS, COLS], torch.float32),
        TensorSpec("y_packed", [ROWS, COLS], torch.float32),
    ]


def spec_expand_clone_batch():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("src", [1, PLANE, PLANE], torch.float32, init_value=_unit([1, PLANE, PLANE])),
        TensorSpec("dst", [BATCH, PLANE, PLANE], torch.float32),
    ]


def spec_expand_dtypes():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("xf", [ROWS, COLS], torch.float16, init_value=_unit([ROWS, COLS])),
        TensorSpec("vf", [ROWS, 1], torch.float16, init_value=_unit([ROWS, 1])),
        TensorSpec(
            "xi",
            [ROWS, COLS],
            torch.int32,
            init_value=lambda: torch.randint(-64, 64, (ROWS, COLS), dtype=torch.int32),
        ),
        TensorSpec(
            "vi",
            [ROWS, 1],
            torch.int32,
            init_value=lambda: torch.randint(-64, 64, (ROWS, 1), dtype=torch.int32),
        ),
        TensorSpec("yf", [ROWS, COLS], torch.float16),
        TensorSpec("yi", [ROWS, COLS], torch.int32),
    ]


SPEC_BUILDERS = {
    "expand_normalize": spec_expand_normalize,
    "expand_softmax": spec_expand_softmax,
    "row_expand_family": _spec_row_family,
    "col_expand_family": _spec_col_family,
    "row_expand_add_tmp": spec_row_expand_add_tmp,
    "expand_clone_batch": spec_expand_clone_batch,
    "expand_dtypes": spec_expand_dtypes,
}


def build_tensor_specs(entry: str = "expand_normalize"):
    """TensorSpec list for *entry*, in that kernel's parameter order."""
    return SPEC_BUILDERS[entry]()


# ---------------------------------------------------------------------------
# Golden references. The row/col family goldens are plain numpy-style
# broadcasts; the two bare expands are copies of the carrier, which is why
# their golden does not mention x at all.
# ---------------------------------------------------------------------------
def golden_expand_normalize(tensors):
    x = tensors["x"]
    row_lo = x.min(dim=-1, keepdim=True).values
    row_hi = x.max(dim=-1, keepdim=True).values
    unit = (x - row_lo) / (row_hi - row_lo)
    tensors["y"][:] = unit * tensors["gamma"] + tensors["beta"]


def golden_expand_softmax(tensors):
    import torch

    x = tensors["x"]
    shifted = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    tensors["y"][:] = shifted / shifted.sum(dim=-1, keepdim=True)


def _fill_row_family(tensors, x, vec):
    import torch

    tensors["o_bare"][:] = vec.expand(ROWS, COLS)
    tensors["o_add"][:] = x + vec
    tensors["o_sub"][:] = x - vec
    tensors["o_mul"][:] = x * vec
    tensors["o_div"][:] = x / vec
    tensors["o_max"][:] = torch.maximum(x, vec)
    tensors["o_min"][:] = torch.minimum(x, vec)
    tensors["o_expdif"][:] = torch.exp(x - vec)


def golden_row_expand_family(tensors):
    _fill_row_family(tensors, tensors["x"], tensors["rv"])


def golden_col_expand_family(tensors):
    _fill_row_family(tensors, tensors["x"], tensors["cv"])


def golden_row_expand_add_tmp(tensors):
    tensors["y_vec"][:] = tensors["x"] + tensors["rv"]
    tensors["y_packed"][:] = tensors["x"] + tensors["rp"].repeat(1, COLS // PACK)


def golden_expand_clone_batch(tensors):
    tensors["dst"][:] = tensors["src"].repeat(BATCH, 1, 1)


def golden_expand_dtypes(tensors):
    tensors["yf"][:] = tensors["xf"] * tensors["vf"]
    tensors["yi"][:] = tensors["xi"] + tensors["vi"]


ENTRIES = [
    # name, kernel, golden, rtol, atol
    ("expand_normalize", expand_normalize, golden_expand_normalize),
    ("expand_softmax", expand_softmax, golden_expand_softmax),
    ("row_expand_family", row_expand_family, golden_row_expand_family),
    ("col_expand_family", col_expand_family, golden_col_expand_family),
    ("row_expand_add_tmp", row_expand_add_tmp, golden_row_expand_add_tmp),
    ("expand_clone_batch", expand_clone_batch, golden_expand_clone_batch),
    ("expand_dtypes", expand_dtypes, golden_expand_dtypes),
]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("-k", "--only", type=str, default=None, help="run one entry by name")
    args = parser.parse_args()

    failures = 0
    for name, fn, golden_fn in ENTRIES:
        if args.only and args.only != name:
            continue
        print(f"\n===== {name} =====", flush=True)
        result = run(
            fn=fn,
            specs=build_tensor_specs(name),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device,
                        enable_chip_swimlane=args.enable_chip_swimlane),
            rtol=1e-5,
            atol=1e-5,
        )
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall broadcast-expand entries passed")