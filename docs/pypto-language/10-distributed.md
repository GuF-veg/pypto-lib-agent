# Distributed Kernels

Multi-chip PyPTO kernels are written in a second namespace, `pld`, alongside the
single-chip `pl` language. The model is: a **host orchestrator** launches the
same kernel on several devices, each device gets a **window** of shared memory,
and devices communicate either by remote read/write or by collectives.

Distributed execution is real and in production use in this repository — but,
as the last section explains, not through the API the older documentation
advertises.

## The `pld` namespace

```python
import pypto.language as pl
import pypto.language.distributed as pld
```

`pld` is an alias for `pypto.language.distributed`, and the alias name matters:
the parser recognises distributed constructs by the literal text `pld.`, so
binding the module to any other name silently breaks dispatch. Use `pld`.

Exports:

| Group | Names |
|---|---|
| Buffers | `alloc_window_buffer`, `window` |
| Topology | `world_size`, `rank`, `nranks`, `get_comm_ctx` |
| Remote access | `remote_load`, `remote_store` |
| Types | `DistributedTensor`, `CommCtx` |
| Enums | `AtomicType`, `NotifyOp`, `ReduceOp`, `WaitCmp` |
| Submodules | `pld.tensor`, `pld.system`, `pld.tile` |

**There is no `pld.allreduce`.** Only eight names exist in the two-segment short
form; the collectives are three-segment, under `pld.tensor`:

```python
pld.tensor.allreduce(...)      pld.tensor.barrier(...)
pld.tensor.broadcast(...)      pld.tensor.allgather(...)
pld.tensor.all_to_all(...)     pld.tensor.all_to_all_v(...)
pld.tensor.reduce_scatter(...)
```

## `DistributedTensor` and windows

`DistributedTensor` is an **annotation-only** type. It takes the same
`[shape, dtype, ...]` subscript as `pl.Tensor`, and differs only in the IR
object kind it produces.

```python
T, D = 4096, 512


@pl.jit.host
def launch(x: pl.Tensor[[T, D], pl.BF16], y: pl.Out[pl.Tensor[[T, D], pl.BF16]]):
    buf = pld.alloc_window_buffer([T, D], dtype=pl.BF16)
    win: pld.DistributedTensor[[T, D], pl.BF16] = pld.window(buf, [T, D], dtype=pl.BF16)
    for r in pl.range(pld.world_size()):
        kernel(x, win, y, device=r)
    return y
```

- `pld.alloc_window_buffer(size_or_shape, dtype=...)` allocates the shared
  buffer. It is **host-only** (calling it inside an InCore body is a parse
  error: "must appear as the RHS of a simple assignment"), `name=` is not
  accepted, and it is an **assignment-intercepted** construct: the parser
  requires its result to be bound to a named variable.
- `pld.window(buf, shape, dtype=...)` produces the `DistributedTensor` view of
  the buffer. It is an ordinary call with no assignment requirements; in host
  code the idiomatic (but optional) spelling is the annotated assignment, which
  is what the parser-generated IR round-trips through.

## Topology: rank and world size

`CommCtx` has **no attributes** — you cannot write `ctx.rank`. Read topology
through functions:

```python
r  = pld.system.rank(ctx)        # -> INT32
n  = pld.system.nranks(ctx)      # -> INT32
ws = pld.world_size()            # -> INT64, host-only
```

There is no group name: a communication domain is an **inferred set of device
ids**, never declared by the author.

## The host orchestrator

There is no `FunctionType.HOST`. `@pl.jit.host` specializes to
`@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)`. The host
orchestrator:

- owns `alloc_window_buffer`, `window` and `world_size()`;
- runs the `for r in pl.range(pld.world_size())` launch loop;
- may pass `device=r` to a callee — legal only when the callee's role is
  `Orchestrator`.

In production models this decorator appears dozens of times.

## Launching

```python
from pypto.runtime import DistributedConfig

run(
    fn=decode_fwd,
    specs=build_tensor_specs(),
    golden_fn=golden_decode,
    config=dict(
        platform="a2a3",
        distributed_config=DistributedConfig(device_ids=[0, 1, 2, 3]),
    ),
)
```

- The number of ranks is **exactly `len(device_ids)`**. There is no `--nranks`
  or `--rank` flag anywhere; the world size is passed to the generated entry as
  a keyword argument.
- `ir.compile` auto-detects a distributed program (any function at Linqu level
  >= 3) and returns a `DistributedCompiledProgram` instead of a
  `CompiledProgram`.
- Window buffers are **compiler-generated**: you pass ordinary host torch
  tensors in parameter order, and the runtime materialises the windows.
  `DistributedConfig` also takes `num_sub_workers`, `runtime` and
  `aicpu_thread_num`.
- Per-script flags vary: models commonly add `--ep` for the world size and
  assert it matches `N_RANKS`.

### Resident parameters

An L3 launch dispatches per rank, so each slice must already sit on the card
that consumes it. The golden harness exposes this as
`golden.TensorSpec(..., resident=...)` (a harness parameter, not a language
feature): an `int` worker id keeps a tensor whole on that card (matching the
callee's `device=`), and `"stacked"` shards `[world_size, *tail]` so slice `i`
lands on card `i` — the canonical
`for r in range(world_size): child(x[r], device=r)` form. `True` is rejected
as ambiguous; residency is L3-only, uploads once, skips the per-dispatch copies.

## Remote access and notification

| Op | Purpose |
|---|---|
| `pld.remote_load(...)` / `pld.remote_store(...)` | short-form read/write of a peer's window |
| `pld.tensor.put(...)` | push a local window/tensor into a peer's window (GM-to-GM TPUT) |
| `pld.tensor.get(...)` | pull a peer's window into the local one (TGET twin of `put`) |
| `pld.tensor.remote_store(...)` | push a computed tensor value into a peer's window, no GM round trip |
| `pld.system.notify(...)` / `wait(...)` | signal and await a peer |
| `pld.system.defer_wait(...)` | register a wait to be satisfied later |
| `pld.tile.remote_load(...)` / `remote_store(...)` | tile-level remote access inside a device region |
| `pld.tile.put(...)` / `get(...)` | tile-level staging variants of `put`/`get` |

`NotifyOp`, `ReduceOp` and `WaitCmp` parameterise the notification and
reduction behaviour.

The production MoE path hand-rolls its all-to-all from per-destination
`pld.tensor.put` calls plus monotonic-epoch `notify`/`wait` pairs. This is the
dominant distributed pattern in this repository.

## Collectives: measured on two cards

The seven `pld.tensor` collectives were **all run and validated on two 910B4
devices** (`-p a2a3`, September 2026) by
`examples/language/distributed_collectives.py` — mesh and ring allreduce,
broadcast, allgather, reduce_scatter, all_to_all and barrier, each with a torch
golden. Point-to-point `put` / `get` / `remote_store` are validated the same way
by `examples/language/distributed_transfer.py`. The contracts that matter:

- **Rebind idiom.** Every collective returns the window and you assign it back
  to the same name: `pub = pld.tensor.allreduce(pub, sig, op=pld.ReduceOp.Sum)`.
  `barrier` is no exception — `sig = pld.tensor.barrier(sig)`.
- **Signal shapes (InCore rail).** mesh uses `[nranks, 1]`; ring uses
  `[2*(NR-1), NR]` with `NR` a compile-time constant. One signal buffer must
  not be shared between the two conventions.
- **Signal reuse.** The InCore rail's credit barrier is self-clearing, so
  back-to-back calls — including inside `for`/`while`/`if` — can share one
  signal. The HOST builtin allreduce is not self-clearing and is rejected
  inside loops.
- **`mode="ring"` on the host builtin is Sum + FP32 only**, and the ring
  schedule caps at 16 ranks; omitting `signal=` (host synthesis) is mesh-only.
- **`allgather(local, target, sig)`** takes this rank's `[1, SIZE]` chunk as a
  plain tensor; a loaded tile is rejected. Rows of the gathered window are
  local only on their own rank — read peer rows back with
  `pld.tile.remote_load`.
- **`all_to_all(input, target, sig)`** takes the plain `[NR, SIZE]` tensor
  parameter directly on the InCore rail (a window there raises "input must be
  TensorType"); `input[dest, :]` is the chunk destined for rank `dest`, and
  `target[src, :]` comes back holding the chunk from rank `src`.
- **`all_to_all_v(input, target, sig, send_counts, recv_counts)`** is the
  variable-count form (source-verified, not yet exercised by
  `examples/language/`). `send_counts` is this rank's INT32 `[NR]` (or `[NR, 1]`)
  rows-per-destination vector; `recv_counts` is an INT32 `[NR, 1]` window the
  collective writes — `recv_counts[src, 0]` holds how many rows `src` actually
  sent here (clamped to the per-peer capacity `MAX_RECV = target.shape[0] //
  NR`), which is exactly how many were transferred, so bound the read-back loop
  by it. Two semantics changed in the `2f892f9..ee49fcea` window:
  - **`send_counts` must be window-bound on the builtin rails.** Each rank
    stages its own vector into its window and every peer reads the ONE word it
    needs with a single non-cacheable scalar read; the `[NR]` INT32 vector is
    the whole buffer requirement. Stage it from a plain tensor first (a
    `pl.read`/`pl.write` loop), like every other window operand of a
    collective.
  - **The barrier `signal` is credit-based, not self-clearing.** Each call adds
    +1 per round to every peer's slot (wait thresholds 1, then 2) and subtracts
    2 per local slot at the end — never a reset — so back-to-back calls on the
    same windows stay ordered, but a second call must still be ordered after
    the first call's local consumer (the receive window is overwritten in
    place), and calls nested in `for`/`while` loops are rejected by the
    compiler. Zero-initialise the signal once and do not reset it.
  Trim to `recv_counts` **before** arithmetic over the capacity block: the
  untouched tail is uninitialised and may decode as NaN or Inf, which would
  propagate into otherwise-valid rows. Mask first, then compute.
- **Host-launch discipline.** Kernel parameters annotated `pl.Out` follow the
  InOut rule: inside a launch loop pass a fresh slice per rank (`outputs[r]`);
  re-passing the same variable is rejected by `InOutUseDiscipline`.
- A runtime crash in this area surfaces as the generic `507018` — an
  out-of-range `peer=` (e.g. `nranks` itself) stalls the scheduler into a
  timeout rather than failing validation.

Remaining limits, source-verified: the host-builtin ring path really is
Sum + FP32 (`CheckSupportedSumFp32BuiltinVariant`); the ring kernel services at
most 16 ranks; `pld.tile.put`/`get` wrappers exist and are callable (the older
claim that they cannot be authored was not reproduced), and ordinary tensor
ops' tolerance of `DistributedTensor` windows varies op by op — elementwise ops
accept them, so test before assuming a rejection.

## A correction to the older documentation

The developer documentation for distributed ops states that ring allreduce
supports FP16 and the Max/Min/Prod reductions. For the **host-builtin ring
path** that is wrong — the deducer only admits Sum with FP32
(`CheckSupportedSumFp32BuiltinVariant`), while the InCore rail takes Sum, Max,
Min and Prod. The same document over-generalises loop legality to host-side
allreduce; its op table, however, covers all fifteen remote/collective ops the
current source exposes, and no missing ops were found on re-check.

## See also

- [Programming Model](01-programming-model.md) — the `HOST` level and
  `@pl.jit.host`.
- [Compiling and Running](09-compiling-and-running.md) — `RunConfig` and
  `DistributedConfig`.
- [L3 Programming](../pypto-coding/l3-programming.md) — the window-buffer /
  notify-wait protocol in full: lanes, epochs, `defer_wait`, and the scheduling
  rules.