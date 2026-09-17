# Patterns and Pitfalls

This page collects what production kernels in this repository actually do, the
mistakes that cost the most time, and the specific places where older
documentation contradicts the implementation.

## The canonical kernel file

Every kernel file here follows the same skeleton. Deviating from it costs you
the lint checks and makes the file harder to compare with its neighbours.

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One line naming the operator, then the formula it computes and how it is tiled."""
import pypto.language as pl

ROWS = 512          # batch / sequence length
HIDDEN = 512        # hidden dimension
ROW_TILE = 64       # rows per core-group
HIDDEN_TILE = 64    # columns per sequential chunk


@pl.jit
def rms_norm(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_norm_rows"):
            ...
    return y


def build_tensor_specs(rows: int = ROWS, hidden: int = HIDDEN):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [rows, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, hidden], torch.float32),
    ]


def golden_rms_norm(tensors):
    import torch

    x = tensors["x"]
    gamma = tensors["gamma"]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    tensors["y"][:] = x / rms * gamma


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    result = run(
        fn=rms_norm,
        specs=build_tensor_specs(),
        golden_fn=golden_rms_norm,
        config=dict(platform=args.platform, device_id=args.device,
                    enable_chip_swimlane=args.enable_chip_swimlane),
        rtol=1e-2,
        atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
```

Details that are conventions rather than requirements, but are followed
everywhere: the spec builder is named `build_tensor_specs` (only `hello_world.py`
uses `build_specs`), the reference is named `golden_<entry_fn>`, every tiling
constant carries a comment saying what it is sized by, and `-p` defaults to
`a2a3` rather than a simulator.

## `examples/` and `models/` are two dialects

This is the single most useful thing to know before reading this repository.
The `examples/` tree teaches the *language*; the `models/` tree shows how
production kernels are *built*, and the two barely overlap.

| Construct | `examples/` | `models/` |
|---|---|---|
| `pl.spmd` | 0 (one stale comment) | 600+ |
| `deps=[...]` | 0 | 470+ |
| `pl.pipeline(stage=)` | 0 | 300+ |
| `pl.unroll` | 0 | 100+ |
| `pl.dynamic` / `bind_dynamic` | 0 | 300+ / 1100+ |
| `pl.tensor.dim` | 0 | 400+ |
| `pl.Out` / `pl.InOut` | few | `pl.InOut` 550+ |
| `pld.*` | one file | 2000+ |
| `@pl.jit.inline` / `.host` | 1 / 1 | 350+ / 47 |
| `pl.while_`, `pl.cond`, `pl.cluster`, `pl.graph`, `pl.submit` | 0 | 0 |

**A guide built from `examples/` alone cannot produce a model kernel.** Read
`examples/` to learn syntax, then read a model under `models/` before writing
anything serious.

## Production patterns

### Three decorator tiers

Production code uses the decorators as a layering system:

- `@pl.jit` — the orchestration entry the harness calls;
- `@pl.jit.inline` — a reusable body, spliced into each caller;
- `@pl.jit.incore` — a standalone worker kernel;
- `@pl.jit.host` — the multi-chip launcher;
- `@pl.jit.extern` — a hand-written CCE kernel.

The orchestrator typically takes 60-90 parameters (the DeepSeek decode layers
do) and performs **no arithmetic at all** — it sequences stage calls and returns.

### Fan out with `pl.spmd`, chain with `deps=`

The canonical production shape is a grid dispatch whose block index selects the
work, chained to the next stage by a captured task id:

```python
with pl.scope():
    with pl.spmd(blocks, name_hint="q_proj") as t0:
        kernel_a(x, y)
    with pl.spmd(blocks, name_hint="epilogue", deps=[t0]) as t1:
        kernel_b(y, z)
```

Inside the kernel, the block index comes from `pl.tile.get_block_idx()`, rows are
tiled with `pl.range`, and the K dimension is walked with
`pl.pipeline(..., stage=2)`.

### Split-K

Two idioms appear:

- a second reduction pass over the partials;
- `pl.assemble(y, partial, offsets, atomic=pl.AtomicType.Add)`.

Prefer the reduction pass when determinism matters; the atomic form is
non-deterministic by contract.

### Matmul accumulate

`pl.matmul_acc(acc, a, b, init_cond=(k == 0))` is the dominant form (about 100
sites). The minority alternative is an explicit
`if k == 0: pl.matmul(...) else: pl.matmul_acc(...)`, which is worth using when
the `init_cond` expression gets complicated.

### Tiling constants

Module-level `ALL_CAPS` constants derived from the model config, with a
divisibility assert nearby. Runtime dimensions stay out of tile sizes.

### Naming

`name_hint` is a short snake_case stage name; there are 680 distinct hints in
`models/`. They are how you find a region in a profile, so do not leave them
empty and do not reuse one for two different stages.

## Pitfalls

### Syntax

| Symptom | Cause |
|---|---|
| `Unsupported expression type: IfExp` | ternary; use an `if`/`else` that yields |
| `Unsupported function call: helper(2)` | calling a plain Python function from a body |
| `Augmented assignment is not supported` | `x += 1`; write `x = x + 1` |
| `Cannot retrieve source code for function` | the kernel is not in a real file (REPL, `exec`, string) |
| `Only simple comparisons supported` | chained comparison; split with `and` |
| `Slice step is not supported` | `x[0:64:2]`; write a loop |
| `Unknown keyword argument '<k>' in pl.range()` | `pl.range` accepts only `init_values` |

### Types and annotations

| Symptom | Cause |
|---|---|
| `Unknown type in subscript: pl.Out` | `pl.Out` used as a return annotation; it is parameter-only |
| `Incomplete type annotation: pl.TaskId` | use `pl.Scalar[pl.TASK_ID]` |
| `Layout variable 'BUF' must be a TensorLayout, got MemRef` | a `MemRef` variable in a tensor's third slot; inline it |
| `Unknown operation 'pl.MemRef'` | a `MemRef` declared inside a body; declare it outside |
| `-> None` rejected | omit the return arrow |
| output reads back as zeros | the tensor was not declared `pl.Out` |

### Loops and control flow

| Symptom | Cause |
|---|---|
| `Mismatch: N iter_args but M init_values` | the carried-name tuple does not match `init_values` |
| `ForStmt YieldStmt value count (3) != iter_args count (2)` | over-yielding; extras are dropped silently until `SSAVerify` |
| post-loop name is undefined | the result is the **yield's** left-hand name, not the header name |
| `pl.unroll() requires compile-time constant integer bounds` | `pl.unroll` needs constants and rejects `init_values` |
| `if my_int:` rejected | conditions must be `Bool` scalars; there is no truthiness |

### Ops and memory

| Symptom | Cause |
|---|---|
| `cannot mix Tensor and Tile arguments` | one binary op mixing levels |
| `acc M=... != matmul M=...` | matmul tile shapes disagree |
| a `Mat` tile fails to store | `pl.store` accepts `Vec` or `Acc` sources only |
| `div` shape error where `mul` worked | `div` does not broadcast; the shapes must match exactly |
| `pl.max` returns something odd | `pl.max` is a **scalar** op, not a reduction |

### Running

| Symptom | Cause |
|---|---|
| `cannot run compiler 'g++-15'` | simulator run without the toolchain shim on `PATH` |
| `Backend type already set to Ascend910B, cannot change to Ascend950` | two architectures in one process |
| every call recompiles | the persistent cache is enabled and silently bypassed here; leave it off |
| edited generated C++ has no effect | use `python -m pypto.runtime.debug.replay <work_dir>` |
| `import golden` fails | run from the repository root with `PYTHONPATH="$PWD"` |
| chip swimlane level is 1 instead of 4 | pass `enable_chip_swimlane` as an explicit integer |

## Where older documentation is wrong

These are the specific contradictions found while writing this guide. Treat the
named claims as unreliable.

| Claim in older docs | Reality |
|---|---|
| `pl` is the only accepted module alias | any alias works; the parser resolves by origin module |
| `pl.range(..., scope=...)` exists | no such keyword (`language/scope.py:40` is stale) |
| `pl.range` accepts `pipeline_stages=`, `unroll=`, `kind=` | accepts only `init_values` |
| `iter_args=` is the DSL keyword | it is IR-only; the DSL spells it `init_values=` + `pl.yield_` |
| `pl.cond` is a multi-branch/predicate helper | it sets the `pl.while_` loop condition only |
| the parser auto-inserts `pl.at` / InCore scopes | it does not; only *runtime* scopes are inserted, by a pass |
| `pl.create_tensor` takes `memory_space=` / `valid_shape=` | neither parameter exists; `init_value=` was removed |
| `pl.syncall`, `pl.fence`, `pl.cacheinvalid` are top-level | they exist only as `pl.system.<name>` |
| `pl.ci` exists | it is `pl.arange`, or `pl.tile.ci` |
| `pl.DN` is a usable layout marker | rejected; use `pl.TensorView(stride=..., layout=pl.DN)` |
| shape expressions must be single scalar variables | composite `DynVar` arithmetic lowers correctly |
| the pass list has 52 entries | there are 54; `NormalizeStmtStructure` is missing from the docs |
| ring allreduce supports FP16 and Max/Min/Prod | Sum + FP32 only, 4-byte elements |
| `tests/golden` passing proves the compiler works | it can pass against a stub pypto; CI never installs pypto |
| the DSL enforces SSA | the default is non-SSA reassignment, type-stable |
| `x.shape` is generally usable | it is `@pl.jit`-only; a hard error under `@pl.function` |
| `pl.submit` is the way to dispatch | it is a parser construct that is never spelled here; use `pl.at`/`pl.spmd` with `deps=` |
| error codes `E001`-`E305` identify failures | the codes are dead code; nothing raises or prints them |

### Corrections established by device verification

The rows above were found by reading implementations and documentation against
each other. The rows below were confirmed by **running kernels on a real Ascend
910B4** (`-p a2a3`); each is backed by a report under
`.pypto-guide-verify/reports/` and a runnable kernel under
`examples/language/`. Several of them are wrong in a way that produces a wrong
answer rather than an error, so they are worth reading even if you never touch
the older pages.

| Claim you may still meet | What the device shows |
|---|---|
| "These ops broadcast numpy-style; `div` is the exception" | Backwards. Mixing `[64,64]` with `[1,64]` is accepted and then **silently miscomputes** for `add`, `sub`, `mul`, `maximum`, `minimum`, `shl`, `shr` (add: 8064/8192 elements wrong; only row 0 is right). `div`, `rem`, `fmod` and the bitwise ops are **rejected** — the safe failure. Use `pl.col_expand`/`pl.row_expand_*`. |
| `pl.mul(tile, [1, C])` is fine | It compiles, runs, and is **wrong** (16127/16384 mismatched, row 0 accidentally correct). Use `pl.col_expand_mul`. |
| `pl.sub(rowvec, tile)` subtracts the tile from the vector | It silently computes `tile - rowvec`. Use `pl.row_expand_sub`. |
| `pl.rsqrt(x)` is accurate | It is a **fast approximation**: about 3.3e-3 relative error (~27000 ULP), systematically failing any 1e-5 gate. Use `pl.rsqrt(x, high_precision=True)` (~1.9e-7). This is why `examples/intermediate/rms_norm.py` ships `rtol=1e-2`. |
| `pl.matmul_acc` works like an accumulate-into-zero | Without `init_cond` it accumulates into **stale L0C**, giving 4096/4096 wrong. Always pass `init_cond=(k == 0)`. |
| `pl.transpose(t)` swaps the last two axes | The call requires **two** axes: `pl.transpose(t, 0, 1)`. |
| `pl.concat(a, b, axis=...)` concatenates on an axis | No `axis=` argument; it always concatenates on the **last** axis and requires identical dtypes. |
| The tensor path allows one narrowing cast | No such limit: `FP32 -> BF16 -> FP16 -> FP32` compiles and runs correctly. |
| `pl.reshape` can reshape/transpose freely | It is a **zero-copy row-major re-view** and must preserve the element count. Proven by a mutually exclusive test: the `x.reshape(64,32)` golden passes while `x.T` fails 2046/2048. |
| Every reduction takes a scratch `tmp_tile` | The family is split with no pattern to infer: `row_*` and `col_sum`/`col_argmax` take it; **`col_max`, `col_min`, `col_prod` take `input` only**. `pl.col_sum(t, tmp)` compiles, `pl.col_max(t, tmp)` gives `col_max() takes 1 positional argument but 2 were given`. |
| `pl.aiv_shard(t, split=1)` / `pl.aic_gather(t, split=1)` take a split mode | Inside a `pl.split_aiv` loop the mode is **inherited** and passing `split=` is a parse error. Outside such a region they are invalid entirely (`must appear inside a pl.split_aiv region`), and the operand must be an `Acc` matmul result - a `Vec` operand is rejected with `drop the pl.aiv_shard and let the implicit affinity-gated split halve it`. |
| `pl.aiv_shard` splits any tile in half | It is specifically the `Acc -> Vec` cube-to-vector crossing; a `Vec` operand is rejected. **The hand-written positive form is established** (September 2026): under `for _ in pl.spmd(1)` with a `pl.split_aiv` region, `acc = pl.matmul(...)` followed by `v = pl.aiv_shard(acc)` and `m = pl.aic_gather(v2)` **inside the region** lowers and validates — see `examples/language/cross_core_shard.py`. Earlier failures came from feeding it `Mat` loads instead of an `Acc` operand. |
| `pl.fillpad_expand(t, [R, C])` broadcasts the row | It **fills**. A `[1, C]` tile widened to `[32, C]` keeps the data in row 0 and puts `pad_value` in rows 1..31 - `0.0` by default, `+inf` with `PadValue.max`. Use `pl.row_expand` to copy a row down. |
| `pad_value=` takes any number | Only `zero`, `max`, `min` (or `0`, `0.0`, `math.inf`, `-math.inf`). `pad_value=7.0` gives `fillpad pad_value only accepts the float literals 0.0, math.inf, or -math.inf; got 7.0`. |
| `pl.get_subblock_idx()` returns a usable number | It yields an **index** scalar, and `pl.cast(k, target_type=pl.FP32)` is rejected with `Cast between float and index`. |
| `pl.move(t, pl.MemorySpace.Mat)` relocates a tile in place | A **cross-space** move is a mixed-kernel transfer, not a local copy: it fails with `'pto.tpush' op tile type must map to a supported producer pipe`. Only same-space moves stand alone. Same cross-core pipe as hand-written `tpush`. |
| `pl.slice` takes `offsets=` | The parameter is **`offset`**, singular. `pl.slice(t, shape, offsets=[0, 0])` fails with `slice() got an unexpected keyword argument 'offsets'`. The positional order is `(input, shape, offset)` - shape before offset. |
| `pl.row_argmax(t, tmp)` returns the max, or a value/index pair | It returns **one Tile holding the index as raw bits reinterpreted into FP32** - a row whose max sits at column 21 stores `2.942726775082116e-44`, the FP32 bit pattern of the integer 21. Reinterpret with `pl.reinterpret_view` to read it. Same convention as `pl.sort32`'s index lanes. |
| `pl.row_sum([R, C])` returns something you reshape later | It returns **`[R, 1]`** — last axis collapsed, dimension kept. `pl.col_sum` mirrors it with **`[1, C]`**. |
| A slice is how you load a tile | `x[a:b, c:d]` yields a **Tensor** subview; only `pl.load` yields a **Tile**, and the levels cannot be mixed. The `tile = x[r:r+TILE,:]` / `y[...] = tile` idiom lowers to a tensor `assemble` and never touches load/store. |
| `pl.create_tensor` produces a tile | It produces a **Tensor**, which is why `pl.store` rejects it. It also has no `memory_space=`, no `valid_shape=`, and `init_value=` was removed. |
| `pl.gather(src, indices, axis=...)` | The axis is spelled **`dim=`**; index shape equals output shape, index rank equals input rank, dtype INT32. |
| `pl.mgather` / `pl.mscatter` are masked | `m` means **mem**, not mask — neither has a mask operand. `pl.mscatter` is **unusable on A2/A3**: it compiles, runs, and silently stores nothing. |
| Hand-written `pl.tpush_to_aiv(t)` is a drop-in for the compiler's pair | `split=` is **required** (`tpush_to_aiv() missing 1 required keyword-only argument: 'split'`). Fixing that only starts a chain: the pair collides with the auto-inserted pipe, then demands a matching `initialize_pipe` with the same `id=`, then matching `import_peer_buffer` calls, then reports **too many** `reserve_buffer` calls, then demands a concrete SSA for `c2v_consumer_buf` rather than the default `0`. Six distinct failures; the counts are checked per expanded function while a `@pl.jit` body is one merged function. Let the compiler emit the pipe, buffers and pair. |
| `pl.sort32(src, idx, tmp=...)` takes a scratch tile | It takes **exactly two operands**, `pl.sort32(src, idx)` - `tmp=` belongs to `pl.mrgsort`. Passing it fails with `sort32() got an unexpected keyword argument 'tmp'`. |
| `pl.sort32` returns two outputs | It returns **one** interleaved pair output; index lanes are raw UINT32 bits reinterpreted into FP32. Its docstring advice to seed indices `[0,2,4,...]` is stale — use consecutive indices. |
| `pl.mrgsort`'s `block_len` counts pairs | It counts tile **elements**: `block_len=64` means runs of 32 values. |
| `pl.scatter` is not a real instruction | The **index form** is a genuine `pto.tscatter`, verified on device; the old wording describes only the mask form. |
| `pl.paged_gather` is Cube-core only | True only for the default `space=Mat` path; `space=pl.MemorySpace.Vec` works in an ordinary core-group region. |
| `pl.tmov_x2zz` is just an A5 variant of `pl.move` | It needs **raw `UINT8`** operands (`requires raw UINT8 src/tmp`), and on a2a3 it has **no codegen at all** (`No codegen registered for operation: tile.tmov_x2zz`). `.lower()` succeeds, so the frontend tells you nothing. |
| `pl.expands` broadcasts a tile | It is **unusable** — there is no backend codegen (`PartialCodegenError: No codegen registered for operation: tile.expands`). |
| `pl.system.sync_src` / `sync_dst` are an alternative to `sync_set` / `sync_wait` | They have **no codegen either**: `No codegen registered for operation: system.sync_src`. Use `sync_set` / `sync_wait`. |
| `pl.col_argmax` / `pl.col_argmin` give you the max or its index | **Do not use them.** They return one Tile, not a value/index pair, and storing it produces **zeros** — measured on a deterministic `arange(R*C) % 7` input the `[1, C]` row came back `[0.0, ...]` while `amax(dim=0)` is `[6.0, ...]`. Use `pl.col_max` for the value, or `pl.sort32` / `pl.mrgsort` for indices. |
| `pl.set_cache_policy(p, pl.CachePolicy.BYPASS)` is a performance hint | It is a coherency contract, and `BYPASS` **kills the run** — about 27 s in, `RuntimeError: chip run lane is poisoned: finalize_native_run failed with code -100`. `DEFAULT` is a verified no-op. |
| `pl.no_dep(t)` can be passed at a call argument | Rejected — re-confirmed September 2026: `UnsupportedFeatureError: Unsupported function call: k_producer(x, pl.no_dep(mid))`. The `create_tensor` docstring still advertises this spelling; it is stale. Use `no_dep_args=[t]` on `pl.at(...)`, or `manual_dep=True` on `pl.create_tensor` for the coarser form. |
| `pl.adir.input(x)` wraps an argument | The `pl.adir.*` markers are **enum values, not functions** - `pl.adir.input(x)` raises `TypeError: 'ArgDirection' object is not callable`. Pass the bare marker in a list: `attrs={"arg_directions": [pl.adir.input, pl.adir.output]}`. |
| `high_precision=True` improves `div`, `log` and `recip` | Their **defaults are already accurate**: at `rtol=1e-7` on device `pl.div`, `pl.recip` and `pl.log` all pass. `rsqrt` is the only op whose default is materially approximate (3.3e-3). The flag exists on exactly six ops - `div`, `log`, `recip`, `rem`, `fmod`, `rsqrt` - and buying it defensively elsewhere is wasted work. |
| Tile column extent is free to choose | A Vec tile is addressed as a flat byte run, so `cols * sizeof(dtype)` must be a **multiple of 32 bytes**. A `[16, 16]` INT8 tile is rejected (`row byte size ... to be 32-byte aligned, but got 16 bytes`) while `[16, 16]` FP32 (64 bytes) is fine. For INT8 that means 32 columns, not 16. Keep the logical extent free with `valid_shape=`. |
| A default `pl.cast` to INT32 rounds like `torch.round` | `torch.round` is **ties-to-even**; the default cast mode is **half away from zero** (C `round()`). Input `-30.5` gives `-31`, not `-30`. They agree on non-ties, so only half-integer inputs expose it. Use `mode="rint"` for ties-to-even. |
| `pl.prefetch.*` works anywhere in a kernel | It must sit inside a `pl.at(level=pl.Level.CORE_GROUP)` scope, and its operand must be flat logical-1D (`pl.reshape(a, [N])` first). At bare orchestration level the verifier reports `references undefined function 'prefetch.make_context'`. |
| `pl.system.bar_v` / `bar_m` / `bar_all` are interchangeable unit barriers | Only **`bar_v()`** is safe in a vector kernel. Run alone on one identical `pl.spmd(4)` launch: `bar_v` passed, `bar_all` stalled the scheduler (`S1:running-stalled`), `bar_m` poisoned the run (`code -100`). Calling all three crashed the device (`507018`). |
| `pl.adds` / `pl.subs` / `pl.muls` / `pl.divs` exist | Those four are **not exported** (`Unknown operation 'pl.adds'`); write `pl.add(tile, scalar)`. A bare integer literal is re-stamped to the operand dtype, so no `pl.cast` is needed. |
| `pl.rem` and `pl.fmod` are interchangeable | `pl.rem` floors (like `torch.remainder`), `pl.fmod` truncates (like `torch.fmod`) — 0 mismatches against the matching torch op, 1999 against the other. |
| `pl.max` / `pl.min` are reductions | They are **scalar-only** operations; on tiles they fail inside nanobind with a message that never mentions `pl.maximum`. |
| A natural `while` loop can carry state | The loop parses, but mutating a **tile** across iterations is rejected by `ConvertToSSA` (`SSAForm` verification). Carry such state with `pl.range(init_values=)` or `pl.while_`. |
| Composite DynVar arithmetic in a shape is an error | It lowers correctly in a plain `@pl.jit` kernel; the restriction bites when the kernel is **inlined**, because the renamer cannot rewrite DynVars embedded in annotations. Prefer named locals. |
| `pl.full(shape, dtype, value)` may be called positionally | The positional form misdispatches inside the nanobind layer. Production code always writes the keyword form — `pl.full([M, N], dtype=pl.FP32, value=0.0)` — and so should you. |
| `pl.at(level=pl.Level.CORE_GROUP)` may wrap code inside an InCore body | `SplitIncoreOrch` rejects it: `InCore ScopeStmt found in non-InCore function (should have been outlined)`. An InCore body (`@pl.jit.incore`, a `pl.spmd` body) is already device code — write the tile ops directly. |
| `pl.system.set_ffts(ws)` takes any workspace shape | The workspace must be the **whole** tensor — a `INT64` 1-D slice raises `system.set_ffts workspace must be a Tensor` — at least 256 elements, and it is accepted only inside the established cross-core handoff idiom. A `sync_set`/`sync_wait` pair outside that idiom crashed the run with `code -100`. |
| Atomic `store(..., atomic=Add)` combines in any dtype | The hardware set is **fp32/bf16/fp16/int32/int16/int8**; `FP8` is rejected at parse: `requires an fp32/bf16/fp16/int32/int16/int8 tile (hardware atomic-add dtypes), but got fp8e4m3fn`. |
| `pl.tile.random` generates values anywhere | It is **A5-only**: `'pto.trandom' op trandom is only supported for A5 targets`. |
| `pl.tensor.view(x, shape)` re-shapes a tensor freely | The identity view compiles and dispatches, then **crashes the runtime** (`finalize_native_run failed with code 507018`), both identity and reshaped. Reading through a view worked in one probe (`pl.load(view, ...)`); treat write-back as broken. |
| `pl.gather_compare(src, kv, tmp, out_cols=K)` computes match indices | On a2a3 the InCore compilation of the generated kernel fails (`Incore compilation failed with exit code 1`) — no usable form found. The mask-form `pl.tile.gather_mask` / `pl.tile.scatter_mask` work. |

## A debugging order that works

1. **Read the IR.** `fn.lower()` shows the entry plus the outlined `k_incore_*`
   kernels; most "wrong results" are a missing `pl.at`, a wrong slice, or an
   accumulation that never initialised.
2. **Dump the pass pipeline.** `dump_passes=True` gives one file per stage, so
   you can see where an intended construct was rewritten away.
3. **Check the annotation directions.** A missing `pl.Out` produces zeros, not
   an error.
4. **Never widen a tolerance first.** Start strict
   (`ratio_allclose(max_error_ratio=0.0)`), look at the error shape with
   `error_distribution`, and only then relax deliberately.
5. **Check the launch geometry** if a kernel is correct but slow: unused blocks
   are common. A hard `syncall` at partial occupancy is a different animal - the
   `HardSyncallOccupancy` verifier rejects it at compile time rather than letting
   it deadlock at runtime, so it shows up as a build error, not a slow kernel
   ([Tensor and system operations](20-tensor-system-ops.md)).

## See also

- [Syntax](02-syntax.md) — the rejection messages in full.
- [Control Flow](04-control-flow.md) — `init_values` and `pl.yield_` rules.
- [Compiling and Running](09-compiling-and-running.md) — harness and lint rules.