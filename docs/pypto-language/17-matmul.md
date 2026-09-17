# Matrix Multiply

`pl.matmul` and its family are the cube (AIC) path: operands are placed in
L0A/L0B, the product accumulates in L0C, and the result comes back as a tile or
tensor. Every contract here was verified by running
`examples/language/matmul_family.py` on a real Ascend 910B4 with `-p a2a3`
(7 entries, 23 compared outputs, three consecutive clean runs).

## Quick reference

| Operator | Call | Shapes | Notes |
|---|---|---|---|
| `pl.matmul` | `pl.matmul(a, b)` | `a[M,K] @ b[K,N] -> [M,N]` | Tensor or Tile |
| `pl.matmul_acc` | `pl.matmul_acc(acc, a, b, init_cond=...)` | `acc[M,N]`, `a[M,K]`, `b[K,N]` | `init_cond` is mandatory in practice |
| `pl.matmul_bias` | `pl.matmul_bias(a, b, bias)` | bias `[1,N]`, accumulator dtype | added once, to the accumulator |
| `pl.gemv` | `pl.gemv(a, b)` | `a[1,K] @ b[K,N] -> [1,N]` | Tile-only |
| `pl.gemv_acc` | `pl.gemv_acc(acc, a, b, acc_phase=...)` | as above | needs `st_phase=Final` on the store |
| `pl.gemv_bias` | `pl.gemv_bias(a, b, bias)` | bias `[1,N]` | |
| `pl.batch_matmul` | `pl.batch_matmul(a, b)` | `[B,M,K] @ [B,K,N] -> [B,M,N]` | Tile-only; batch is the leading axis |
| `pl.tile.batch_matmul_acc` | `pl.tile.batch_matmul_acc(acc, a, b, init_cond=...)` | `acc += a @ b`, batch dims broadcast like `batch_matmul` | Tile-only and **not promoted to `pl.`** — `acc` must be a Tile (e.g. a `pl.matmul` result), not a Tensor |

## The one thing that will bite you: `init_cond`

```python
acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
for kb in pl.range(K // K_TILE):
    k0 = kb * K_TILE
    acc = pl.matmul_acc(acc, x[mb : mb + M_TILE, k0 : k0 + K_TILE],
                        w[k0 : k0 + K_TILE, nb : nb + N_TILE],
                        init_cond=(kb == 0))
```

**`pl.matmul_acc` without `init_cond` is unusable.** The accumulator is not
zeroed for you: it starts as whatever L0C held, and the run completes with
4096/4096 elements wrong. `init_cond` says "on this iteration, overwrite rather
than accumulate", and it must be a `Bool` scalar. Codegen makes it explicit:

```text
if (v84 == v15) { TMATMUL(...) } else { TMATMUL_ACC(...) }
```

The minority alternative, when the predicate is awkward, is an explicit branch:

```python
if kb == 0:
    acc = pl.matmul(a, b)
else:
    acc = pl.matmul_acc(acc, a, b)
```

`pl.pipeline` needs the explicit `init_values=` / `pl.yield_` pair where
`pl.range` accepts a bare rebinding.

## Transpose flags

| Flag | Meaning |
|---|---|
| `b_trans=False` (default) | right operand stored `[K, N]` |
| `b_trans=True` | right operand stored `[N, K]` |
| `a_trans=True` | left operand stored `[K, M]` |

Both flags are **tensor-only**. On Tile operands they raise, and the message
points you at `pl.tile.transpose_view(...)`, which works on device. `a_trans`
is functional but unused in this repository's kernels; production only needs
`b_trans`.

## dtype rules

- **Operands must have identical dtype**: `ValueError: The operator tile.matmul
  requires identical lhs and rhs data types, but got fp16 and fp32`.
- **`out_dtype` defaults to FP32** for FP32, FP16 and BF16 operands - the cube
  accumulator dtype. Declaring an FP32 output with no `out_dtype` is correct.
- `out_dtype=pl.FP16` and `pl.BF16` work. `out_dtype=pl.INT32` on FP16 inputs is
  rejected: reaching int32 from fp32 is a quantization needing a scale the
  attribute cannot carry.

## `pl.matmul_bias`

`pl.matmul_bias(a, b, bias)` - bias is the **third positional** argument, shape
`[1, N]`, and its dtype must match the **accumulator**, not the inputs: FP32 bias
even for FP16 operands (`requires bias dtype fp32 to match the accumulator, but
got fp16`). It is added **once**, to the accumulator, not once per K block: with
K = 512 forcing internal K blocking, codegen emits `TMATMUL_BIAS` on block 0 and
`TMATMUL_ACC` for blocks 1-3, and three calls produce exactly three
`TMATMUL_BIAS`.

## `pl.gemv` and its traps

`pl.gemv` is Tile-only, requires the left operand to be exactly `[1, K]` with a
`[K, N]` right operand, and produces `[1, N]`. Two hard limits:

- **A one-element extent fails.** `[K, 1]` gets a `DN` layout and `TLOAD`
  rejects it (`only support ND2ND/DN2DN/NZ2NZ/ND2NZ/DN2ZN`); an `[M, 1]` output
  fails `TSTORE`. N = 1 fails; N = 2 and N = 16 pass. A rank-1 `[K]` right
  operand is rejected (`requires rhs to be 2D, but got 1 dimensions`).
- **`gemv_acc`'s accumulator cannot be synthesized.** It must have valid shape
  `[1, N]` and physical shape `[16, N]` at once, so `init_cond` cannot seed a
  fresh accumulator. Use `pl.gemv(acc_phase=pl.AccPhase.Partial)` then
  `pl.gemv_acc(acc_phase=pl.AccPhase.Final)`, and you **must** pass
  `st_phase=pl.STPhase.Final` on the store or compilation fails with
  `Verification failed after 'InlineFunctions' for properties {AccStorePhaseValid}`.

## Memory space

Placement is automatic. `pl.load` accepts only `Vec` or `Mat` - `Left` and
`Right` are rejected (`target_memory for tile.load must be MemorySpace.Vec or
MemorySpace.Mat, got MemorySpace.Left`). The `AutoTileMatmulL0` pass rewrites
`tile.matmul` and friends into a K loop with `tile.extract(..., Left|Right)` and
an accumulator seed; explicit placement via
`pl.move(t, target_memory=Left/Right)` is verified working.

Capacities: L0A and L0B 64 KiB each, L0C 128 KiB, Mat 512 KiB, Bias 1 KiB - so an
FP32 `[64, k]` operand caps `k` at 256.

One trap worth stating plainly: **forcing `target_memory=pl.Mem.Vec` on a matmul
operand is silently accepted and still returns the correct result.** That is the
compiler covering for you, not a licence. The cases that do fail are
`pl.move(..., Vec)` as the right operand (`'pto.tpush' op tile type must map to a
supported pipe`) and `pl.move(..., Acc)` (`'pto.tmov' op expects a supported tmov
address-space pair`).

That first error is **not specific to matmul operands**: any cross-space
`pl.move` hits it, because moving a tile between `Mat` and `Vec` makes the body a
mixed kernel whose transfer needs a cross-core pipe. See
[Tile operations](06-tile-operations.md) for the standalone form and
`examples/language/tile_views.py` for the verified same-space case.

## Tolerances

FP32 outputs use `rtol=atol=1e-4` - the floor the repository's own matmul test
calibrates, with the observed 1e-5 failures being pure FP32 summation-order drift
(worst case about 1.7e-5 absolute, typically 1.1e-5). Narrowed outputs use
ULP-derived bounds: FP16 `1e-3` (one ULP is 2^-11), BF16 `1e-2` (2^-8). No
tolerance was widened to hide a logic failure - the `init_cond` bug above shows up
as 4096/4096 wrong, not as a tolerance cliff.

## Run the examples

```bash
cd <path/to/pypto-lib-agent>   # the repository root
PYTHONPATH="$PWD" conda run -n pypto python examples/language/matmul_family.py -p a2a3 -d 4
```

```text
all matmul-family entries passed          (exit code 0)
```

Entries cover the plain product, the pipelined K loop with `init_cond`, the
transpose flags (with distinct M/N/K so a dropped flag fails on both numbers and
shape), `matmul_bias`, the `gemv` family, `batch_matmul`, and `out_dtype`.
`pl.tile.batch_matmul_acc` was verified separately (September 2026 probe):
starting from a `pl.matmul` accumulator, `init_cond=True` selects the
overwriting step and a following flag-less call accumulates — the two-step
result matched `2 * a @ b` at `rtol=atol=1e-4`.

## See also

- [Tile operations](06-tile-operations.md) - the wider operator set.
- [Data movement](19-data-movement.md) - why `pl.create_tensor` makes a Tensor.
- [Broadcast and expand](16-broadcast-expand.md) - the epilogue ops after a matmul.
