# Scopes and Dependencies

`pl.at` is where orchestration becomes a kernel. Around it sit three related
mechanisms: **runtime scopes** (memory and dependency boundaries),
**explicit dependencies** (patching the task graph), and **per-scope
optimizations** (how the outlined kernel is generated).

## `pl.at` in full

```python
def at(
    level: pl.Level,
    role: pl.Role | None = None,
    *,
    optimizations: list[Optimization] | None = None,
    deps: list[Any] | None = None,
    no_dep_args: list[Any] | None = None,
    dumps: list[Any] | None = None,
    allow_early_resolve: bool = False,
    predicate: Any = None,
    name_hint: str = "",
    windowize: bool = False,
) -> AtContext
```

| Argument | Purpose |
|---|---|
| `level` | lowering target; `pl.Level.CORE_GROUP` is what you use |
| `role` | function role (`Orchestrator` / `SubWorker`) at non-InCore levels; rejected for `CORE_GROUP` |
| `name_hint` | label for the outlined kernel; appears in generated kernel names and traces |
| `optimizations` | per-region codegen hints — `pl.split(...)`, `pl.cross_core_slot(...)` |
| `deps` | explicit producer `TaskId`s this region must wait on |
| `no_dep_args` | captured tensors to exclude from automatic dependency tracking |
| `dumps` | captured tensors to mark for selective dump |
| `allow_early_resolve` | scheduling hint: let consumers pre-stage before this completes |
| `predicate` | dispatch predicate — skip the whole task when false |
| `windowize` | allow local windowization for the outlined kernel (default `False`) |

With `level=pl.Level.CORE_GROUP` and no `optimizations`, the scope lowers to an
InCore scope. With `optimizations=[pl.split(mode)]` it lowers to an InCore scope
carrying the split mode. Any other level produces a hierarchy scope.

`name_hint` must be a valid identifier and should be a short snake_case stage
name (`"q_proj"`, `"rms_norm_rows"`). Production kernels use hundreds of
distinct hints; they are the primary way to find a region in a profile.

You can capture the region's producer task id:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1") as tid:
    y = stage1(x, y)

with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[tid]):
    z = stage2(y, z)
```

## `optimizations`

Two entries exist, and they are orthogonal:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj",
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=4)]):
    ...
```

- **`pl.split(mode)`** splits a mixed cube+vector region so the cube and vector
  units can ping-pong on the two halves. `mode` is `pl.SplitMode.NONE`,
  `UP_DOWN` (halve rows) or `LEFT_RIGHT` (halve columns). It applies only to a
  region that contains both cube and vector work.
- **`pl.cross_core_slot(slot_num=N)`** sets the ring depth of the automatic
  cube-to-vector pipe. It sizes a channel, it does not partition work; raising it
  costs on-chip memory (`slot_size * slot_num`). The default is 2.

`pl.split(mode, slot_num=N)` is the **deprecated** spelling of the pair and
emits a `DeprecationWarning`. Use `pl.cross_core_slot`.

The parser requires these entries to be written **inline as literals** at the
call site — it inspects the AST, so a list built in a variable is not accepted.

## Runtime scopes

A runtime scope is a memory-reclaim and dependency-tracking boundary for the
simpler runtime. It is *not* the same thing as `pl.at`:

```python
with pl.scope():                          # AUTO dependency tracking
    out = kernel(a, b, out)

with pl.scope(mode=pl.ScopeMode.MANUAL):  # you own every dependency edge
    out, tid = pl.submit(stage1, x, out)
```

- **By default the compiler inserts them for you** — for the function body and
  for each `for`/`if` body. Writing them is a tuning mechanism, never a
  correctness requirement.
- To place them yourself, set `@pl.function(auto_scope=False)` (or
  `@pl.jit(auto_scope=False)`) and use `pl.scope(...)`. There is no
  `pl.range(..., scope=...)` sugar: `pl.range` accepts only bounds and
  `init_values=`/`attrs=` (a stale docstring in `language/scope.py` still
  advertises the sugar; the parser rejects the keyword).
- `pl.scope(mode=AUTO)` is only allowed when `auto_scope=False`; `MANUAL` is
  allowed either way. AUTO may not nest inside MANUAL.
- `pl.manual_scope()` is shorthand for `pl.scope(mode=pl.ScopeMode.MANUAL)`.
  Inside it the runtime skips automatic dependency lookup entirely, so you must
  declare every edge yourself.

Beware a naming collision: **`pl.AUTO` is `-1`**, the `reserve_buffer` base
sentinel. It is unrelated to `pl.ScopeMode.AUTO`, which is `0`.

## The dependency model

The runtime tracks dependencies automatically by watching which tasks read and
write which tensors. Four mechanisms let you intervene, from finest to
coarsest:

| Mechanism | Scope of effect |
|---|---|
| `pl.no_dep(t)` at a call argument | one tensor, one task |
| `pl.create_tensor(..., manual_dep=True)` | one tensor, its whole lifetime |
| `deps=[...]` on `pl.at` / `pl.spmd` / `pl.submit` | adds edges on top of automatic ones |
| `pl.manual_scope()` | the entire region |

`deps=` is **additive**: the final dependency set is the automatic set unioned
with your explicit edges, so `deps=` works in auto scope too. Use it as a
precision tool to add an edge the runtime cannot infer, and reach for
`manual_scope` only when you want to own the whole graph.

Dependency edges are named by **task ids**, which are `pl.Scalar[pl.TASK_ID]`
values. They can be captured with `as tid`, returned by `pl.submit` /
`pl.spmd_submit`, stored in a `pl.Array[N, pl.TASK_ID]`, and passed into kernels
as parameters.

## `pl.submit` and `pl.spmd_submit`

```python
out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid])
(a, b), tid = pl.submit(self.multi_out_kernel, x)
out, tid    = pl.spmd_submit(self.incore_kernel, x, y, core_num=8, deps=[t])
```

- Both are **parser constructs, not functions.** Their Python bodies raise if
  called; the parser intercepts the syntax. They exist so that the names resolve
  for imports and linters.
- The result is a 2-tuple: the kernel's result(s) and a producer `TaskId`.
  A single left-hand side is also accepted and binds the whole tuple.
- `core_num=` is required on `pl.spmd_submit`.
- Optional keywords: `deps=[...]`, `allow_early_resolve=True`, `timing_slot=N`,
  `predicate=(...)`. `dumps=[...]` exists on `pl.submit` and `pl.at` only —
  the parser rejects it on `pl.spmd_submit` and kernel-call positions.

In `pypto-lib-agent`, **`pl.submit` is never spelled.** Its behaviour is reached
through `pl.at(..., deps=...)` and the `as tid` forms of `pl.at` and `pl.spmd`,
which lower to the same submit internally. Use those.

## Dispatch predicates

A predicate lets the scheduler skip a task at dispatch time, when the decision
only becomes known at run time:

```python
with pl.spmd(1) as gate_tid:
    row_count = self.gate(row_count)

with pl.spmd(4, deps=[gate_tid], predicate=(row_count[0, 0] > 0)) as tid:
    out = self.expert(x, out)
```

- The comparison is **parsed, never evaluated** in orchestration. Only
  `tensor[indices] OP int-literal` is expressible — one comparison, with `==`,
  `!=`, `>`, `<`, `>=` or `<=`. No chained comparisons, arithmetic or boolean
  combination.
- When false, the task is retired inline: never dispatched to a core, but its
  fan-in and fan-out still settle so consumers unlock.
- **Contract:** the operand tensor's producer must be one of this task's
  `deps=`, otherwise the predicate may read a stale value. The parser checks
  where it can prove the relationship, but getting it right is your job.
- On `pl.at`, `predicate=` is only valid with `level=pl.Level.CORE_GROUP` and
  only outside `pl.cluster()` / `pl.spmd()` / another core-group `pl.at`.

## Scheduling hints

- `allow_early_resolve=True` marks a task as a speculative early-dispatch
  producer: the scheduler may pre-stage its consumers and release them the
  instant it finishes. A consumer pre-stages only once *all* its producers are
  flagged. Purely an optimisation; it pays off on critical paths built from many
  short tasks.
- `timing_slot=N` (on `pl.submit` / `pl.spmd_submit`) associates the task with a
  timing slot for profiling.

## Selective dumping

`dumps=[t1, t2]` on a scope and `pl.submit`'s `dumps=` mark specific tensor
arguments for dump. Each entry must be a tensor passed positionally. These marks
only take effect under partial dump (`RunConfig.enable_dump_args == 1`); they
are a no-op when dumping is off and irrelevant under full dump.
`pl.dump_tag(t)` is the declarative equivalent.

## `pl.cluster` and `pl.graph`

Both exist, both are unused in this repository, and both have sharp edges.

`pl.cluster(*, name_hint="")` groups AIC and AIV kernels for co-scheduled
execution:

```python
with pl.cluster():
    with pl.spmd(4, sync_start=True):
        out = self.mixed_kernel(a, b, out)
```

Inside a cluster, a `pl.spmd` is folded into a group function, so the
submit-only keywords (`deps=`, `allow_early_resolve=`, `predicate=`, `as tid`)
are rejected there.

`pl.graph(name)` marks a recordable orchestration fragment: the
`host_build_graph` runtime records the task topology on the first call and
replays it afterwards. It requires compiling under
`RuntimeKind.HOST_BUILD_GRAPH`, is rejected inside InCore/AIC/AIV/Group/Spmd
functions and inside `pl.at` / `pl.cluster` / `pl.spmd`, and does not accept
`as`.

## Which mechanisms production code actually uses

Counted across `examples/` and `models/` in this repository:

| Mechanism | Uses |
|---|---|
| `deps=[...]` | ~478 |
| `pl.spmd` (`with` / `for` / `as tid`) | ~640 |
| `pl.at` | ~426 (all `level=CORE_GROUP`) |
| `pl.scope()` | ~146 |
| `auto_scope=False` | ~131 |
| `pl.manual_scope()` | ~17 |
| `pl.split_aiv` | ~23 |
| `pl.spmd_submit` | ~4 |
| `pl.cross_core_slot` | ~4 |
| `pl.submit` | 0 |
| `pl.cluster`, `pl.graph` | 0 |

The canonical production shape is a `pl.scope()` around `pl.spmd(...) as tid`
dispatches chained by `deps=[tid]`.

## See also

- [Control Flow](04-control-flow.md) — `pl.spmd` forms in detail.
- [Compiling and Running](09-compiling-and-running.md) — `RunConfig` dump and
  scope options.
- [Patterns and Pitfalls](11-patterns-and-pitfalls.md) — how models chain tasks.