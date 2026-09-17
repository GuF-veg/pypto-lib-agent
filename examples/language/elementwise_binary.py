# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Binary elementwise and bitwise tile operators.

Tile-tile arithmetic: add, sub, mul, div, maximum, minimum, rem, fmod.
Scalar-right-hand-side: adds, subs, muls, divs, maximums, minimums, rems, fmods
(and the ``pl.add(tile, 3)`` auto-dispatch onto the scalar form).
Carry forms: addc, subc, addsc, subsc.
Partial forms: part_add, part_mul, part_max, part_min.
Bitwise / shift on INT32: and_, or_, xor, shl, shr with ands, ors, xors, shls, shrs.
Explicit broadcast: col_expand feeding a same-shape binary op, because implicit
shape broadcasting is type-checked but silently miscomputed on A2/A3.

Tiling: 128x64 FP32/INT32 tensors in 64-row tiles; each pl.parallel block owns a
pl.at InCore scope holding two [64, 64] tiles, one [64, 64] scratch tile for the
``rem``/``xor`` tmp operand, and one output store per operator.
"""
import pypto.language as pl

ROWS = 128
COLS = 64
TILE = 64
HALF = COLS // 2


# ---------------------------------------------------------------------------
# 1. tile-tile arithmetic (FP32)
# ---------------------------------------------------------------------------
@pl.jit
def arith_tile_tile(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    w: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_add: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_sub: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_mul: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_div: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_maximum: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_minimum: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_rem: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_fmod: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="arith_tile_tile"):
            a = pl.tile.load(x, [r, 0], [TILE, COLS])
            b = pl.tile.load(w, [r, 0], [TILE, COLS])
            tmp = pl.tile.create([TILE, COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            pl.tile.store(pl.add(a, b), offsets=[r, 0], output_tensor=y_add)
            pl.tile.store(pl.sub(a, b), offsets=[r, 0], output_tensor=y_sub)
            pl.tile.store(pl.mul(a, b), offsets=[r, 0], output_tensor=y_mul)
            pl.tile.store(pl.div(a, b), offsets=[r, 0], output_tensor=y_div)
            pl.tile.store(pl.maximum(a, b), offsets=[r, 0], output_tensor=y_maximum)
            pl.tile.store(pl.minimum(a, b), offsets=[r, 0], output_tensor=y_minimum)
            pl.tile.store(pl.rem(a, b, tmp), offsets=[r, 0], output_tensor=y_rem)
            pl.tile.store(pl.fmod(a, b), offsets=[r, 0], output_tensor=y_fmod)
    return y_add, y_sub, y_mul, y_div, y_maximum, y_minimum, y_rem, y_fmod


def spec_arith_tile_tile():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        # Divisor magnitude kept in [0.5, 1.5] so div/rem/fmod stay well conditioned.
        TensorSpec("w", [ROWS, COLS], torch.float32,
                   init_value=lambda: (0.5 + torch.rand(ROWS, COLS))
                   * torch.where(torch.rand(ROWS, COLS) < 0.5, -1.0, 1.0)),
        TensorSpec("y_add", [ROWS, COLS], torch.float32),
        TensorSpec("y_sub", [ROWS, COLS], torch.float32),
        TensorSpec("y_mul", [ROWS, COLS], torch.float32),
        TensorSpec("y_div", [ROWS, COLS], torch.float32),
        TensorSpec("y_maximum", [ROWS, COLS], torch.float32),
        TensorSpec("y_minimum", [ROWS, COLS], torch.float32),
        TensorSpec("y_rem", [ROWS, COLS], torch.float32),
        TensorSpec("y_fmod", [ROWS, COLS], torch.float32),
    ]


def golden_arith_tile_tile(t):
    import torch

    x, w = t["x"], t["w"]
    t["y_add"][:] = x + w
    t["y_sub"][:] = x - w
    t["y_mul"][:] = x * w
    t["y_div"][:] = x / w
    t["y_maximum"][:] = torch.maximum(x, w)
    t["y_minimum"][:] = torch.minimum(x, w)
    # pl.rem is the FLOOR remainder (TREM): it tracks torch.remainder, not fmod.
    t["y_rem"][:] = torch.remainder(x, w)
    # pl.fmod is the TRUNCATING remainder (TFMOD): the result follows the dividend.
    t["y_fmod"][:] = torch.fmod(x, w)


# ---------------------------------------------------------------------------
# 2. scalar-right-hand-side arithmetic (FP32)
# ---------------------------------------------------------------------------
@pl.jit
def arith_tile_scalar(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    w: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_adds: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_subs: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_muls: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_divs: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_maximums: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_minimums: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_rems: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_fmods: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_add_tile: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_add_literal: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_adds_explicit: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="arith_tile_scalar"):
            a = pl.tile.load(x, [r, 0], [TILE, COLS])
            b = pl.tile.load(w, [r, 0], [TILE, COLS])
            tmp = pl.tile.create([TILE, COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            pl.tile.store(pl.tile.adds(a, 3), offsets=[r, 0], output_tensor=y_adds)
            pl.tile.store(pl.tile.subs(a, 2), offsets=[r, 0], output_tensor=y_subs)
            pl.tile.store(pl.tile.muls(a, 2), offsets=[r, 0], output_tensor=y_muls)
            pl.tile.store(pl.tile.divs(a, 2), offsets=[r, 0], output_tensor=y_divs)
            pl.tile.store(pl.maximums(a, 0), offsets=[r, 0], output_tensor=y_maximums)
            pl.tile.store(pl.minimums(a, 0), offsets=[r, 0], output_tensor=y_minimums)
            pl.tile.store(pl.rems(a, 2, tmp), offsets=[r, 0], output_tensor=y_rems)
            pl.tile.store(pl.fmods(a, 2), offsets=[r, 0], output_tensor=y_fmods)
            # pl.add(tile, tile) is the tile-tile form; pl.add(tile, 3) auto-dispatches
            # onto tile.adds. The three add outputs must therefore be identical.
            pl.tile.store(pl.add(a, b), offsets=[r, 0], output_tensor=y_add_tile)
            pl.tile.store(pl.add(a, 3), offsets=[r, 0], output_tensor=y_add_literal)
            pl.tile.store(pl.tile.adds(a, 3), offsets=[r, 0], output_tensor=y_adds_explicit)
    return (y_adds, y_subs, y_muls, y_divs, y_maximums, y_minimums, y_rems,
            y_fmods, y_add_tile, y_add_literal, y_adds_explicit)


def spec_arith_tile_scalar():
    import torch

    from golden import TensorSpec

    names = ("y_adds", "y_subs", "y_muls", "y_divs", "y_maximums", "y_minimums",
             "y_rems", "y_fmods", "y_add_tile", "y_add_literal", "y_adds_explicit")
    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("w", [ROWS, COLS], torch.float32, init_value=torch.randn),
        *[TensorSpec(n, [ROWS, COLS], torch.float32) for n in names],
    ]


def golden_arith_tile_scalar(t):
    import torch

    x, w = t["x"], t["w"]
    t["y_adds"][:] = x + 3.0
    t["y_subs"][:] = x - 2.0
    t["y_muls"][:] = x * 2.0
    t["y_divs"][:] = x / 2.0
    t["y_maximums"][:] = torch.maximum(x, torch.zeros_like(x))
    t["y_minimums"][:] = torch.minimum(x, torch.zeros_like(x))
    t["y_rems"][:] = torch.remainder(x, 2.0)
    t["y_fmods"][:] = torch.fmod(x, 2.0)
    t["y_add_tile"][:] = x + w
    t["y_add_literal"][:] = x + 3.0
    t["y_adds_explicit"][:] = x + 3.0


# ---------------------------------------------------------------------------
# 3. carry forms (FP32): op(a, b, carry) == a <op> b + carry
# ---------------------------------------------------------------------------
@pl.jit
def arith_carry(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    w: pl.Tensor[[ROWS, COLS], pl.FP32],
    c: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_addc: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_subc: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_addsc: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_subsc: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="arith_carry"):
            a = pl.tile.load(x, [r, 0], [TILE, COLS])
            b = pl.tile.load(w, [r, 0], [TILE, COLS])
            carry = pl.tile.load(c, [r, 0], [TILE, COLS])
            pl.tile.store(pl.addc(a, b, carry), offsets=[r, 0], output_tensor=y_addc)
            pl.tile.store(pl.subc(a, b, carry), offsets=[r, 0], output_tensor=y_subc)
            pl.tile.store(pl.addsc(a, 3, carry), offsets=[r, 0], output_tensor=y_addsc)
            pl.tile.store(pl.subsc(a, 3, carry), offsets=[r, 0], output_tensor=y_subsc)
    return y_addc, y_subc, y_addsc, y_subsc


def spec_arith_carry():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("w", [ROWS, COLS], torch.float32, init_value=torch.randn),
        # Carry is an ORDINARY per-element addend, not a boolean flag: use 0..3 so a
        # saturating or boolean implementation would disagree with the golden.
        TensorSpec("c", [ROWS, COLS], torch.float32,
                   init_value=lambda: torch.randint(0, 4, (ROWS, COLS)).to(torch.float32)),
        TensorSpec("y_addc", [ROWS, COLS], torch.float32),
        TensorSpec("y_subc", [ROWS, COLS], torch.float32),
        TensorSpec("y_addsc", [ROWS, COLS], torch.float32),
        TensorSpec("y_subsc", [ROWS, COLS], torch.float32),
    ]


def golden_arith_carry(t):
    x, w, c = t["x"], t["w"], t["c"]
    t["y_addc"][:] = x + w + c
    t["y_subc"][:] = x - w + c
    t["y_addsc"][:] = x + 3.0 + c
    t["y_subsc"][:] = x - 3.0 + c


# ---------------------------------------------------------------------------
# 4. partial forms (FP32) on fully-valid sources
# ---------------------------------------------------------------------------
@pl.jit
def arith_partial(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    w: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_part_add: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_part_mul: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_part_max: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_part_min: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="arith_partial"):
            a = pl.tile.load(x, [r, 0], [TILE, COLS])
            b = pl.tile.load(w, [r, 0], [TILE, COLS])
            pl.tile.store(pl.part_add(a, b), offsets=[r, 0], output_tensor=y_part_add)
            pl.tile.store(pl.part_mul(a, b), offsets=[r, 0], output_tensor=y_part_mul)
            pl.tile.store(pl.part_max(a, b), offsets=[r, 0], output_tensor=y_part_max)
            pl.tile.store(pl.part_min(a, b), offsets=[r, 0], output_tensor=y_part_min)
    return y_part_add, y_part_mul, y_part_max, y_part_min


def spec_arith_partial():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("w", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("y_part_add", [ROWS, COLS], torch.float32),
        TensorSpec("y_part_mul", [ROWS, COLS], torch.float32),
        TensorSpec("y_part_max", [ROWS, COLS], torch.float32),
        TensorSpec("y_part_min", [ROWS, COLS], torch.float32),
    ]


def golden_arith_partial(t):
    import torch

    x, w = t["x"], t["w"]
    # With both sources fully valid the partial forms are exactly the plain forms.
    t["y_part_add"][:] = x + w
    t["y_part_mul"][:] = x * w
    t["y_part_max"][:] = torch.maximum(x, w)
    t["y_part_min"][:] = torch.minimum(x, w)


# ---------------------------------------------------------------------------
# 5. partial forms distinguish themselves from plain forms via the valid region
# ---------------------------------------------------------------------------
@pl.jit
def arith_partial_valid_region(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    w: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_part_add: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_part_mul: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_part_max: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_part_min: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="part_valid_region"):
            a = pl.tile.load(x, [r, 0], [TILE, COLS])
            # b's valid region covers only the left half of its [TILE, COLS] buffer.
            # The partial forms must COPY a wherever b is not valid; the plain forms
            # would instead consume b's undefined buffer there.
            #
            # Do NOT call pl.fillpad on b first: fillpad makes the padded region a
            # defined value and the partial forms then behave exactly like the plain
            # ones (measured: both return a + pad).
            b = pl.tile.load(w, [r, 0], [TILE, COLS], valid_shape=[TILE, HALF])
            pl.tile.store(pl.part_add(a, b), offsets=[r, 0], output_tensor=y_part_add)
            pl.tile.store(pl.part_mul(a, b), offsets=[r, 0], output_tensor=y_part_mul)
            pl.tile.store(pl.part_max(a, b), offsets=[r, 0], output_tensor=y_part_max)
            pl.tile.store(pl.part_min(a, b), offsets=[r, 0], output_tensor=y_part_min)
    return y_part_add, y_part_mul, y_part_max, y_part_min


def spec_arith_partial_valid_region():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("w", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("y_part_add", [ROWS, COLS], torch.float32),
        TensorSpec("y_part_mul", [ROWS, COLS], torch.float32),
        TensorSpec("y_part_max", [ROWS, COLS], torch.float32),
        TensorSpec("y_part_min", [ROWS, COLS], torch.float32),
    ]


def golden_arith_partial_valid_region(t):
    import torch

    x, w = t["x"], t["w"]
    xl, wl = x[:, :HALF], w[:, :HALF]
    xr = x[:, HALF:]
    # Left half: both sources valid, so the combine applies. Right half: only a is
    # valid, so the partial form copies a verbatim.
    t["y_part_add"][:] = torch.cat([xl + wl, xr], dim=1)
    t["y_part_mul"][:] = torch.cat([xl * wl, xr], dim=1)
    t["y_part_max"][:] = torch.cat([torch.maximum(xl, wl), xr], dim=1)
    t["y_part_min"][:] = torch.cat([torch.minimum(xl, wl), xr], dim=1)


# ---------------------------------------------------------------------------
# 6a. bitwise and shift, tile-tile, on INT32
# ---------------------------------------------------------------------------
@pl.jit
def bitwise_tile_int32(
    xi: pl.Tensor[[ROWS, COLS], pl.INT32],
    wi: pl.Tensor[[ROWS, COLS], pl.INT32],
    sh: pl.Tensor[[ROWS, COLS], pl.INT32],
    y_and: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_or: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_xor: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_shl: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_shr: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bitwise_tile_int32"):
            a = pl.tile.load(xi, [r, 0], [TILE, COLS])
            b = pl.tile.load(wi, [r, 0], [TILE, COLS])
            s = pl.tile.load(sh, [r, 0], [TILE, COLS])
            tmp = pl.tile.create([TILE, COLS], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            pl.tile.store(pl.and_(a, b), offsets=[r, 0], output_tensor=y_and)
            pl.tile.store(pl.or_(a, b), offsets=[r, 0], output_tensor=y_or)
            pl.tile.store(pl.xor(a, b, tmp), offsets=[r, 0], output_tensor=y_xor)
            pl.tile.store(pl.shl(a, s), offsets=[r, 0], output_tensor=y_shl)
            pl.tile.store(pl.shr(a, s), offsets=[r, 0], output_tensor=y_shr)
    return y_and, y_or, y_xor, y_shl, y_shr


def spec_bitwise_tile_int32():
    import torch

    from golden import TensorSpec

    names = ("y_and", "y_or", "y_xor", "y_shl", "y_shr")
    return [
        # Non-negative operands keep shl/shr free of signed-shift ambiguity.
        TensorSpec("xi", [ROWS, COLS], torch.int32,
                   init_value=lambda: torch.randint(1, 4096, (ROWS, COLS), dtype=torch.int32)),
        TensorSpec("wi", [ROWS, COLS], torch.int32,
                   init_value=lambda: torch.randint(1, 4096, (ROWS, COLS), dtype=torch.int32)),
        TensorSpec("sh", [ROWS, COLS], torch.int32,
                   init_value=lambda: torch.randint(0, 8, (ROWS, COLS), dtype=torch.int32)),
        *[TensorSpec(n, [ROWS, COLS], torch.int32) for n in names],
    ]


def golden_bitwise_tile_int32(t):
    import torch

    xi, wi, sh = t["xi"], t["wi"], t["sh"]
    t["y_and"][:] = torch.bitwise_and(xi, wi)
    t["y_or"][:] = torch.bitwise_or(xi, wi)
    t["y_xor"][:] = torch.bitwise_xor(xi, wi)
    t["y_shl"][:] = torch.bitwise_left_shift(xi, sh)
    t["y_shr"][:] = torch.bitwise_right_shift(xi, sh)


# ---------------------------------------------------------------------------
# 6b. bitwise and shift, scalar rhs, on INT16.
# On A2/A3 ptoas accepts tands/tors/txors only for i8/i16, so the scalar logical
# forms need a 16-bit tile; the scalar SHIFT forms also accept INT32 (entry 6c).
# ---------------------------------------------------------------------------
@pl.jit
def bitwise_scalar_int16(
    xi: pl.Tensor[[ROWS, COLS], pl.INT16],
    y_ands: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT16]],
    y_ors: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT16]],
    y_xors: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT16]],
    y_shls: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT16]],
    y_shrs: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT16]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bitwise_scalar_int16"):
            a = pl.tile.load(xi, [r, 0], [TILE, COLS])
            tmp = pl.tile.create([TILE, COLS], dtype=pl.INT16, target_memory=pl.MemorySpace.Vec)
            pl.tile.store(pl.ands(a, 255), offsets=[r, 0], output_tensor=y_ands)
            pl.tile.store(pl.ors(a, 15), offsets=[r, 0], output_tensor=y_ors)
            pl.tile.store(pl.xors(a, 5, tmp), offsets=[r, 0], output_tensor=y_xors)
            pl.tile.store(pl.shls(a, 2), offsets=[r, 0], output_tensor=y_shls)
            pl.tile.store(pl.shrs(a, 2), offsets=[r, 0], output_tensor=y_shrs)
    return y_ands, y_ors, y_xors, y_shls, y_shrs


def spec_bitwise_scalar_int16():
    import torch

    from golden import TensorSpec

    names = ("y_ands", "y_ors", "y_xors", "y_shls", "y_shrs")
    return [
        TensorSpec("xi", [ROWS, COLS], torch.int16,
                   init_value=lambda: torch.randint(1, 200, (ROWS, COLS), dtype=torch.int16)),
        *[TensorSpec(n, [ROWS, COLS], torch.int16) for n in names],
    ]


def golden_bitwise_scalar_int16(t):
    import torch

    xi = t["xi"]
    t["y_ands"][:] = torch.bitwise_and(xi, 255)
    t["y_ors"][:] = torch.bitwise_or(xi, 15)
    t["y_xors"][:] = torch.bitwise_xor(xi, 5)
    t["y_shls"][:] = torch.bitwise_left_shift(xi, 2)
    t["y_shrs"][:] = torch.bitwise_right_shift(xi, 2)


# ---------------------------------------------------------------------------
# 6c. scalar SHIFT forms on INT32 (accepted by ptoas, unlike ands/ors/xors)
# ---------------------------------------------------------------------------
@pl.jit
def shift_scalar_int32(
    xi: pl.Tensor[[ROWS, COLS], pl.INT32],
    y_shls: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_shrs: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="shift_scalar_int32"):
            a = pl.tile.load(xi, [r, 0], [TILE, COLS])
            pl.tile.store(pl.shls(a, 2), offsets=[r, 0], output_tensor=y_shls)
            pl.tile.store(pl.shrs(a, 2), offsets=[r, 0], output_tensor=y_shrs)
    return y_shls, y_shrs


def spec_shift_scalar_int32():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("xi", [ROWS, COLS], torch.int32,
                   init_value=lambda: torch.randint(1, 4096, (ROWS, COLS), dtype=torch.int32)),
        TensorSpec("y_shls", [ROWS, COLS], torch.int32),
        TensorSpec("y_shrs", [ROWS, COLS], torch.int32),
    ]


def golden_shift_scalar_int32(t):
    import torch

    xi = t["xi"]
    t["y_shls"][:] = torch.bitwise_left_shift(xi, 2)
    t["y_shrs"][:] = torch.bitwise_right_shift(xi, 2)


# ---------------------------------------------------------------------------
# 7. integer remainder semantics on negative dividends.
# pl.fmod/pl.fmods are FP32-only on A2/A3 (ptoas rejects an INT32 tile.fmod with
# "tile.fmod with dtype int32 is not supported on A2/A3"), so only the floor form
# pl.rem/pl.rems is exercised for INT32 here.
# ---------------------------------------------------------------------------
@pl.jit
def int_remainder_negative(
    xi: pl.Tensor[[ROWS, COLS], pl.INT32],
    wi: pl.Tensor[[ROWS, COLS], pl.INT32],
    y_rem: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_rems: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="int_remainder"):
            a = pl.tile.load(xi, [r, 0], [TILE, COLS])
            b = pl.tile.load(wi, [r, 0], [TILE, COLS])
            tmp = pl.tile.create([TILE, COLS], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            pl.tile.store(pl.rem(a, b, tmp), offsets=[r, 0], output_tensor=y_rem)
            pl.tile.store(pl.rems(a, 3, tmp), offsets=[r, 0], output_tensor=y_rems)
    return y_rem, y_rems


def spec_int_remainder_negative():
    import torch

    from golden import TensorSpec

    return [
        # Signed dividends; A2/A3 TREM requires every INT32 element in [-2**24, 2**24].
        TensorSpec("xi", [ROWS, COLS], torch.int32,
                   init_value=lambda: torch.randint(-1000, 1000, (ROWS, COLS), dtype=torch.int32)),
        TensorSpec("wi", [ROWS, COLS], torch.int32,
                   init_value=lambda: torch.randint(1, 9, (ROWS, COLS), dtype=torch.int32)),
        TensorSpec("y_rem", [ROWS, COLS], torch.int32),
        TensorSpec("y_rems", [ROWS, COLS], torch.int32),
    ]


def golden_int_remainder_negative(t):
    import torch

    xi, wi = t["xi"], t["wi"]
    # Integer conventions differ exactly as they do for floats: torch.remainder is
    # FLOOR (Python //, %), torch.fmod is TRUNCATED toward zero (C %). pl.rem matches
    # the FLOOR convention, verified on device for tile-tile and tile-scalar INT32.
    t["y_rem"][:] = torch.remainder(xi, wi)
    t["y_rems"][:] = torch.remainder(xi, 3)


# ---------------------------------------------------------------------------
# 8. the supported way to broadcast: expand explicitly, then use a same-shape op
# ---------------------------------------------------------------------------
@pl.jit
def explicit_broadcast(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    rv: pl.Tensor[[1, COLS], pl.FP32],
    cv: pl.Tensor[[ROWS, 1], pl.FP32],
    y_col_expand_add: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_row_expand_add: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_col_expand_mul: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="explicit_broadcast"):
            a = pl.tile.load(x, [r, 0], [TILE, COLS])
            row = pl.tile.load(rv, [0, 0], [1, COLS])
            col = pl.tile.load(cv, [r, 0], [TILE, 1])
            # col_expand materialises the [1, COLS] carrier into a full [TILE, COLS]
            # tile, so the following add is an ordinary same-shape tile add.
            pl.tile.store(pl.add(a, pl.tile.col_expand(a, row)), offsets=[r, 0],
                          output_tensor=y_col_expand_add)
            # row_expand_add folds the [TILE, 1] carrier expansion into the add.
            pl.tile.store(pl.row_expand_add(a, col), offsets=[r, 0],
                          output_tensor=y_row_expand_add)
            pl.tile.store(pl.col_expand_mul(a, row), offsets=[r, 0],
                          output_tensor=y_col_expand_mul)
    return y_col_expand_add, y_row_expand_add, y_col_expand_mul


def spec_explicit_broadcast():
    import torch

    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("rv", [1, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("cv", [ROWS, 1], torch.float32, init_value=torch.randn),
        TensorSpec("y_col_expand_add", [ROWS, COLS], torch.float32),
        TensorSpec("y_row_expand_add", [ROWS, COLS], torch.float32),
        TensorSpec("y_col_expand_mul", [ROWS, COLS], torch.float32),
    ]


def golden_explicit_broadcast(t):
    x, rv, cv = t["x"], t["rv"], t["cv"]
    t["y_col_expand_add"][:] = x + rv
    t["y_row_expand_add"][:] = x + cv
    t["y_col_expand_mul"][:] = x * rv


# ---------------------------------------------------------------------------
# 9. scalar-literal dtype stamping on an INT32 tile.
# A bare integer literal is re-stamped to the tile dtype, so neither pl.cast nor
# pl.const is needed. A FLOAT literal on an integer tile is NOT re-stamped: it
# stays FP32 and the hardware narrows it by dropping the fraction, so +1.5 acts
# as +1. Only pl.tile.adds exists for the tile scalar form; pl.adds is not a
# public name (use pl.add(tile, scalar) or pl.tile.adds).
# ---------------------------------------------------------------------------
@pl.jit
def dtype_scalar_int32(
    xi: pl.Tensor[[ROWS, COLS], pl.INT32],
    y_add_int_lit: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_adds_int_lit: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_add_float_lit: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
    y_muls_int_lit: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT32]],
):
    for r in pl.parallel(0, ROWS, TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dtype_scalar_int32"):
            a = pl.tile.load(xi, [r, 0], [TILE, COLS])
            pl.tile.store(pl.add(a, 3), offsets=[r, 0], output_tensor=y_add_int_lit)
            pl.tile.store(pl.tile.adds(a, 3), offsets=[r, 0], output_tensor=y_adds_int_lit)
            pl.tile.store(pl.add(a, 1.5), offsets=[r, 0], output_tensor=y_add_float_lit)
            pl.tile.store(pl.tile.muls(a, 2), offsets=[r, 0], output_tensor=y_muls_int_lit)
    return y_add_int_lit, y_adds_int_lit, y_add_float_lit, y_muls_int_lit


def spec_dtype_scalar_int32():
    import torch

    from golden import TensorSpec

    names = ("y_add_int_lit", "y_adds_int_lit", "y_add_float_lit", "y_muls_int_lit")
    return [
        TensorSpec("xi", [ROWS, COLS], torch.int32,
                   init_value=lambda: torch.randint(0, 1000, (ROWS, COLS), dtype=torch.int32)),
        *[TensorSpec(n, [ROWS, COLS], torch.int32) for n in names],
    ]


def golden_dtype_scalar_int32(t):
    xi = t["xi"]
    t["y_add_int_lit"][:] = xi + 3
    t["y_adds_int_lit"][:] = xi + 3
    # 1.5 is narrowed to the integer tile by dropping the fraction.
    t["y_add_float_lit"][:] = xi + 1
    t["y_muls_int_lit"][:] = xi * 2


ENTRIES = [
    ("arith_tile_tile", arith_tile_tile, spec_arith_tile_tile, golden_arith_tile_tile),
    ("arith_tile_scalar", arith_tile_scalar, spec_arith_tile_scalar, golden_arith_tile_scalar),
    ("arith_carry", arith_carry, spec_arith_carry, golden_arith_carry),
    ("arith_partial", arith_partial, spec_arith_partial, golden_arith_partial),
    ("arith_partial_valid_region", arith_partial_valid_region,
     spec_arith_partial_valid_region, golden_arith_partial_valid_region),
    ("bitwise_tile_int32", bitwise_tile_int32, spec_bitwise_tile_int32,
     golden_bitwise_tile_int32),
    ("bitwise_scalar_int16", bitwise_scalar_int16, spec_bitwise_scalar_int16,
     golden_bitwise_scalar_int16),
    ("shift_scalar_int32", shift_scalar_int32, spec_shift_scalar_int32,
     golden_shift_scalar_int32),
    ("dtype_scalar_int32", dtype_scalar_int32, spec_dtype_scalar_int32,
     golden_dtype_scalar_int32),
    ("int_remainder_negative", int_remainder_negative,
     spec_int_remainder_negative, golden_int_remainder_negative),
    ("explicit_broadcast", explicit_broadcast, spec_explicit_broadcast, golden_explicit_broadcast),
]


def build_tensor_specs(name: str = "arith_tile_tile"):
    """Specs for one named entry (default: the first)."""
    for entry_name, _fn, spec_fn, _golden in ENTRIES:
        if entry_name == name:
            return spec_fn()
    raise KeyError(f"unknown entry {name!r}")


def golden_entry(tensors):
    """Golden dispatcher kept for the single-entry skeleton shape."""
    golden_arith_tile_tile(tensors)


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=1)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("-k", "--only", type=str, default=None, help="run one entry by name")
    args = parser.parse_args()

    failures = 0
    for name, fn, _spec_fn, golden_fn in ENTRIES:
        if args.only and args.only != name:
            continue
        print(f"\n===== {name} =====", flush=True)
        try:
            result = run(
                fn=fn,
                specs=build_tensor_specs(name),
                golden_fn=golden_fn,
                config=dict(platform=args.platform, device_id=args.device,
                            enable_chip_swimlane=args.enable_chip_swimlane),
                rtol=1e-5,
                atol=1e-5,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures += 1
            print(f"{name}: RAISED {type(exc).__name__}: {str(exc).strip()[:800]}", flush=True)
            continue
        if not result.passed:
            failures += 1
            if result.error:
                print(result.error)
    if failures:
        print(f"\n{failures} entry/entries FAILED")
        raise SystemExit(1)
    print("\nall binary-elementwise entries passed")