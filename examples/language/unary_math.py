# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Math unary operators - exp / log / sqrt / rsqrt / recip / abs / neg and their accuracy.

Two entries, each applying its operators to one [ROW_TILE, COL_TILE] block
inside a single CORE_GROUP scope:

    unary_math        x in [0.5, 1.5) and signed z - the ordinary range
    unary_math_wide   exp over [-8, 8] and log/sqrt/rsqrt/recip over eight
                      decades [1e-4, 1e4) - proves the tolerance is not an
                      artifact of a narrow input range

``recip(sqrt(x))`` sits next to ``rsqrt(x)`` so the two can be compared against
the same reference, and ``rsqrt(x, high_precision=True)`` sits next to the
default ``rsqrt(x)`` because the default is a fast approximation that is only
accurate to ~3e-3 relative on a2a3.

The ``probe`` comparator is an honest ``torch.allclose`` gate - it never widens
the tolerance for the accurate operators. It additionally prints, per output,
the worst per-element deviation from the fp32 golden and from a float64 exact
reference, the bitwise-equal count, and the fraction of elements above each
relative-error threshold: the numbers a programmer needs to pick a tolerance.
"""
import pypto.language as pl

ROWS = 256
COLS = 256
ROW_TILE = 64
COL_TILE = 64

# Input ranges.
POS_LO = 0.5
POS_HI = 1.5
WIDE_SIGNED_LO = -8.0
WIDE_SIGNED_HI = 8.0
WIDE_POS_LO = 1e-4
WIDE_POS_HI = 1e4
NEAR_ONE_HALF_WIDTH = 1e-3

# Justified gate for the *default* (fast, approximate) pl.rsqrt only.
# Measured worst-case relative deviation from the fp32 golden on a2a3 is
# 3.28e-03, so 4e-3 is the smallest round tolerance with headroom. Every other
# output is gated at the strict rtol/atol given on the command line.
RSQRT_FAST_RTOL = 4e-3
RSQRT_FAST_ATOL = 0.0

MEASURED: dict = {}


@pl.jit
def unary_math(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    z: pl.Tensor[[ROWS, COLS], pl.FP32],
    xn: pl.Tensor[[ROWS, COLS], pl.FP32],
    y_exp: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_log: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_sqrt: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_rsqrt: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_rsqrt_hp: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_recip: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_recip_sqrt: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_abs: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_neg: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    y_log_near1: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        for c in pl.range(0, COLS, COL_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="unary_math"):
                xp = x[r : r + ROW_TILE, c : c + COL_TILE]
                zt = z[r : r + ROW_TILE, c : c + COL_TILE]
                nt = xn[r : r + ROW_TILE, c : c + COL_TILE]
                y_exp[r : r + ROW_TILE, c : c + COL_TILE] = pl.exp(xp)
                y_log[r : r + ROW_TILE, c : c + COL_TILE] = pl.log(xp)
                y_sqrt[r : r + ROW_TILE, c : c + COL_TILE] = pl.sqrt(xp)
                # Default: fast approximate reciprocal-sqrt (~3e-3 relative).
                y_rsqrt[r : r + ROW_TILE, c : c + COL_TILE] = pl.rsqrt(xp)
                # Optional high-precision lowering (allocates a scratch tile).
                y_rsqrt_hp[r : r + ROW_TILE, c : c + COL_TILE] = pl.rsqrt(xp, high_precision=True)
                y_recip[r : r + ROW_TILE, c : c + COL_TILE] = pl.recip(xp)
                y_recip_sqrt[r : r + ROW_TILE, c : c + COL_TILE] = pl.recip(pl.sqrt(xp))
                y_abs[r : r + ROW_TILE, c : c + COL_TILE] = pl.abs(zt)
                y_neg[r : r + ROW_TILE, c : c + COL_TILE] = pl.neg(zt)
                # x near 1.0: log(x) -> 0, where a relative tolerance is meaningless.
                y_log_near1[r : r + ROW_TILE, c : c + COL_TILE] = pl.log(nt)
    return (
        y_exp,
        y_log,
        y_sqrt,
        y_rsqrt,
        y_rsqrt_hp,
        y_recip,
        y_recip_sqrt,
        y_abs,
        y_neg,
        y_log_near1,
    )


@pl.jit
def unary_math_wide(
    xe: pl.Tensor[[ROWS, COLS], pl.FP32],
    xw: pl.Tensor[[ROWS, COLS], pl.FP32],
    w_exp: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    w_log: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    w_sqrt: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    w_rsqrt: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    w_rsqrt_hp: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    w_recip: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    w_recip_sqrt: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        for c in pl.range(0, COLS, COL_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="unary_math_wide"):
                et = xe[r : r + ROW_TILE, c : c + COL_TILE]
                wt = xw[r : r + ROW_TILE, c : c + COL_TILE]
                w_exp[r : r + ROW_TILE, c : c + COL_TILE] = pl.exp(et)
                w_log[r : r + ROW_TILE, c : c + COL_TILE] = pl.log(wt)
                w_sqrt[r : r + ROW_TILE, c : c + COL_TILE] = pl.sqrt(wt)
                w_rsqrt[r : r + ROW_TILE, c : c + COL_TILE] = pl.rsqrt(wt)
                w_rsqrt_hp[r : r + ROW_TILE, c : c + COL_TILE] = pl.rsqrt(wt, high_precision=True)
                w_recip[r : r + ROW_TILE, c : c + COL_TILE] = pl.recip(wt)
                w_recip_sqrt[r : r + ROW_TILE, c : c + COL_TILE] = pl.recip(pl.sqrt(wt))
    return w_exp, w_log, w_sqrt, w_rsqrt, w_rsqrt_hp, w_recip, w_recip_sqrt


OUTPUTS = [
    "y_exp",
    "y_log",
    "y_sqrt",
    "y_rsqrt",
    "y_rsqrt_hp",
    "y_recip",
    "y_recip_sqrt",
    "y_abs",
    "y_neg",
    "y_log_near1",
]

WIDE_OUTPUTS = [
    "w_exp",
    "w_log",
    "w_sqrt",
    "w_rsqrt",
    "w_rsqrt_hp",
    "w_recip",
    "w_recip_sqrt",
]


def build_tensor_specs(rows: int = ROWS, cols: int = COLS):
    import torch
    from golden import TensorSpec

    def positive() -> "torch.Tensor":
        return torch.rand(rows, cols, dtype=torch.float32) * (POS_HI - POS_LO) + POS_LO

    specs = [
        # x: strictly positive, so log / sqrt / rsqrt / recip never see 0 or a negative.
        TensorSpec("x", [rows, cols], torch.float32, init_value=positive),
        # z: signed, and pushed away from 0 so neg/abs are exercised on both signs.
        TensorSpec(
            "z",
            [rows, cols],
            torch.float32,
            init_value=lambda: torch.randn(rows, cols, dtype=torch.float32) + 0.5,
        ),
        # xn: within +-1e-3 of 1.0, so log(xn) lands close to 0.
        TensorSpec(
            "xn",
            [rows, cols],
            torch.float32,
            init_value=lambda: torch.empty(rows, cols, dtype=torch.float32).uniform_(
                1.0 - NEAR_ONE_HALF_WIDTH, 1.0 + NEAR_ONE_HALF_WIDTH
            ),
        ),
    ]
    specs.extend(TensorSpec(name, [rows, cols], torch.float32) for name in OUTPUTS)
    return specs


def build_wide_specs(rows: int = ROWS, cols: int = COLS):
    import math

    import torch
    from golden import TensorSpec

    def wide_signed() -> "torch.Tensor":
        return torch.empty(rows, cols, dtype=torch.float32).uniform_(WIDE_SIGNED_LO, WIDE_SIGNED_HI)

    def wide_positive() -> "torch.Tensor":
        # Log-uniform over eight decades, so log() spans about [-9.2, 9.2].
        return torch.exp(
            torch.empty(rows, cols, dtype=torch.float32).uniform_(math.log(WIDE_POS_LO), math.log(WIDE_POS_HI))
        )

    specs = [
        TensorSpec("xe", [rows, cols], torch.float32, init_value=wide_signed),
        TensorSpec("xw", [rows, cols], torch.float32, init_value=wide_positive),
    ]
    specs.extend(TensorSpec(name, [rows, cols], torch.float32) for name in WIDE_OUTPUTS)
    return specs


def golden_unary_math(values):
    import torch

    x = values["x"]
    z = values["z"]
    values["y_exp"][:] = torch.exp(x)
    values["y_log"][:] = torch.log(x)
    values["y_sqrt"][:] = torch.sqrt(x)
    values["y_rsqrt"][:] = torch.rsqrt(x)
    values["y_rsqrt_hp"][:] = torch.rsqrt(x)
    values["y_recip"][:] = 1.0 / x
    values["y_recip_sqrt"][:] = 1.0 / torch.sqrt(x)
    values["y_abs"][:] = torch.abs(z)
    values["y_neg"][:] = -z
    values["y_log_near1"][:] = torch.log(values["xn"])


def golden_unary_math_wide(values):
    import torch

    xe = values["xe"]
    xw = values["xw"]
    values["w_exp"][:] = torch.exp(xe)
    values["w_log"][:] = torch.log(xw)
    values["w_sqrt"][:] = torch.sqrt(xw)
    values["w_rsqrt"][:] = torch.rsqrt(xw)
    values["w_rsqrt_hp"][:] = torch.rsqrt(xw)
    values["w_recip"][:] = 1.0 / xw
    values["w_recip_sqrt"][:] = 1.0 / torch.sqrt(xw)


# Exact float64 reference per output, used only for reporting (never for gating).
EXACT = {
    "y_exp": lambda v: v["x"].double().exp(),
    "y_log": lambda v: v["x"].double().log(),
    "y_sqrt": lambda v: v["x"].double().sqrt(),
    "y_rsqrt": lambda v: 1.0 / v["x"].double().sqrt(),
    "y_rsqrt_hp": lambda v: 1.0 / v["x"].double().sqrt(),
    "y_recip": lambda v: 1.0 / v["x"].double(),
    "y_recip_sqrt": lambda v: 1.0 / v["x"].double().sqrt(),
    "y_abs": lambda v: v["z"].double().abs(),
    "y_neg": lambda v: -v["z"].double(),
    "y_log_near1": lambda v: v["xn"].double().log(),
    "w_exp": lambda v: v["xe"].double().exp(),
    "w_log": lambda v: v["xw"].double().log(),
    "w_sqrt": lambda v: v["xw"].double().sqrt(),
    "w_rsqrt": lambda v: 1.0 / v["xw"].double().sqrt(),
    "w_rsqrt_hp": lambda v: 1.0 / v["xw"].double().sqrt(),
    "w_recip": lambda v: 1.0 / v["xw"].double(),
    "w_recip_sqrt": lambda v: 1.0 / v["xw"].double().sqrt(),
}

THRESHOLDS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def probe(name: str, tol: tuple[float, float] | None = None):
    """Gate *name* with ``torch.allclose`` and print its accuracy numbers.

    ``tol`` is an explicit, pre-justified ``(rtol, atol)`` for an operator whose
    hardware accuracy cannot meet the strict gate; without it the caller's
    strict ``rtol``/``atol`` are used unchanged.
    """
    import torch

    def cmp(actual, expected, *, inputs=None, rtol=1e-5, atol=1e-5, **_):
        eff_rtol, eff_atol = tol if tol is not None else (rtol, atol)
        a = actual.double()
        e = expected.double()
        exact = EXACT[name](inputs) if inputs else e
        n = actual.numel()

        def stats(ref):
            finite = torch.isfinite(a) & torch.isfinite(ref)
            if not finite.any():
                return float("inf"), float("inf"), None
            d = (a - ref).abs()[finite]
            denom = ref.abs()[finite]
            return float(d.max()), float((d / denom).max()), d / denom

        max_abs, max_rel, rel_all = stats(e)
        max_abs_x, max_rel_x, _ = stats(exact)
        bit_equal = int((actual == expected).sum())
        fracs = " ".join(f"{t:g}:{float((rel_all > t).double().mean()):.2%}" for t in THRESHOLDS)
        MEASURED[name] = {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "max_abs_exact": max_abs_x,
            "max_rel_exact": max_rel_x,
            "bit_equal": bit_equal,
            "n": n,
            "gate_rtol": eff_rtol,
            "gate_atol": eff_atol,
            "justified": tol is not None,
        }
        print(
            f"[CMP] {name:<14} vs fp32 golden: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"| vs fp64 exact: max_abs={max_abs_x:.3e} max_rel={max_rel_x:.3e} "
            f"| bitwise_equal={bit_equal}/{n} ({bit_equal / n:.1%})"
        )
        print(f"[CMP] {name:<14} min_rtol@atol=0: {max_rel:.3e}   frac(rel>thd): {fracs}")
        if not bool(torch.isfinite(actual).all()):
            print(f"[CMP] {name:<14} NON-FINITE values present in the device output")
        ok = bool(torch.allclose(actual, expected, rtol=eff_rtol, atol=eff_atol))
        tag = " JUSTIFIED-OVERRIDE" if tol is not None else ""
        print(f"[CMP] {name:<14} gate rtol={eff_rtol:g} atol={eff_atol:g} -> {'PASS' if ok else 'FAIL'}{tag}")
        detail = (
            f"  '{name}' max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"bitwise_equal={bit_equal}/{n} rtol={eff_rtol} atol={eff_atol}\n"
        )
        return ok, detail

    cmp.__name__ = f"probe:{name}"
    return cmp


def build_compare_fns(names, overrides=None):
    overrides = overrides or {}
    return {name: probe(name, tol=overrides.get(name)) for name in names}


def print_tolerance_table(title, names):
    print("")
    print(f"[TABLE] {title}")
    print("[TABLE] operator          max_rel_vs_fp32  max_rel_vs_fp64  bitwise_equal   gate_rtol  gate_atol")
    for name in names:
        m = MEASURED.get(name)
        if not m:
            continue
        print(
            f"[TABLE] {name:<18} {m['max_rel']:<16.3e} {m['max_rel_exact']:<16.3e} "
            f"{m['bit_equal']:>6}/{m['n']:<8} {m['gate_rtol']:<10g} {m['gate_atol']:g}"
        )
    print("")


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--rtol", type=float, default=1e-5, help="strict relative tolerance gate")
    parser.add_argument("--atol", type=float, default=1e-5, help="strict absolute tolerance gate")
    parser.add_argument(
        "--rsqrt-fast-rtol",
        type=float,
        default=RSQRT_FAST_RTOL,
        help="justified rtol for the default (fast, approximate) pl.rsqrt",
    )
    parser.add_argument(
        "--strict-rsqrt",
        action="store_true",
        help="gate the fast pl.rsqrt at the strict rtol too (expected to FAIL), proving the measurement",
    )
    args = parser.parse_args()

    fast_tol = None if args.strict_rsqrt else (args.rsqrt_fast_rtol, RSQRT_FAST_ATOL)
    print(f"[RUN] strict gate: rtol={args.rtol:g} atol={args.atol:g}")
    if fast_tol:
        print(f"[RUN] pl.rsqrt (fast path) gate override: rtol={fast_tol[0]:g} atol={fast_tol[1]:g}")
    else:
        print("[RUN] pl.rsqrt (fast path) gated at the strict rtol as well")

    config = dict(
        platform=args.platform,
        device_id=args.device,
        enable_chip_swimlane=args.enable_chip_swimlane,
    )

    print("[RUN] === entry unary_math: x in [0.5, 1.5), z signed ===")
    narrow = run(
        fn=unary_math,
        specs=build_tensor_specs(),
        golden_fn=golden_unary_math,
        config=config,
        rtol=args.rtol,
        atol=args.atol,
        compare_fn=build_compare_fns(OUTPUTS, {"y_rsqrt": fast_tol}),
    )
    print_tolerance_table("unary_math (narrow range)", OUTPUTS)

    print("[RUN] === entry unary_math_wide: exp over [-8,8], log/recip over [1e-4,1e4) ===")
    wide = run(
        fn=unary_math_wide,
        specs=build_wide_specs(),
        golden_fn=golden_unary_math_wide,
        config=config,
        rtol=args.rtol,
        atol=args.atol,
        compare_fn=build_compare_fns(WIDE_OUTPUTS, {"w_rsqrt": fast_tol}),
    )
    print_tolerance_table("unary_math_wide (wide range)", WIDE_OUTPUTS)

    for result in (narrow, wide):
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
    print("[RUN] PASS - every operator within its documented tolerance")