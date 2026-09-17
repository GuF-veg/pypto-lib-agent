# Control Flow

PyPTO has a small, closed set of loop constructs. Choosing between them is a
**semantic** decision, not a stylistic one: each one tells the compiler what
scheduling and code generation are valid.

| Construct | Kind in IR | Meaning |
|---|---|---|
| `pl.range` | `Sequential` | iterations run in order; loop-carried state allowed |
| `pl.parallel` | `Parallel` | iterations are independent and may be spread across cores |
| `pl.unroll` | `Unroll` | sequential, replicated at compile time |
| `pl.pipeline` | `Pipeline` | sequential, software-pipelined across units |
| `while cond:` | `WhileStmt` | data-dependent loop (natural form) |
| `pl.while_` | `WhileStmt` | data-dependent loop with explicit carried state |
| `pl.spmd` | `SpmdScopeStmt` | SPMD dispatch of a block-parallel body |
| `pl.split_aiv` | `SplitAivScopeStmt` | explicit AIV split region |

The kind is chosen by **which helper you call**; the enum that carries it is
`pl.ForKind` (`Sequential`, `Parallel`, `Unroll`, `Pipeline`). There is no `kind=` argument
anywhere.

## Argument shape

`pl.range`, `pl.parallel`, `pl.unroll` and `pl.pipeline` share Python's `range`
calling convention:

```text
pl.<loop>(stop)
pl.<loop>(start, stop)
pl.<loop>(start, stop, step)
```

Each argument may be a Python `int` or a `pl.Scalar`. Be aware of two things the
API does **not** have, even though some documentation claims otherwise:
`pl.range` accepts no `scope=`, `pipeline_stages=`, `unroll=` or `kind=`
keyword, and there is no `iter_args=` spelling — loop-carried values use
`init_values=` (below).

## `pl.range` — sequential

```python
for kb in pl.range(K // K_TILE):            # 0 .. K//K_TILE
for kb in pl.range(1, K // K_TILE):         # 1 .. K//K_TILE
for k0 in pl.range(0, K, K_TILE):           # start, stop, step
```

Use it for the K-reduction loop of a matmul, for a sequential walk over column
tiles inside one `pl.at` region, and for any loop whose iterations must observe
each other's writes.

## `pl.parallel` — independent iterations

```python
for mb in pl.parallel(0, M, M_TILE):
    for nb in pl.parallel(0, N, N_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tile"):
            ...
```

`pl.parallel` marks iterations as independent, which is what lets the compiler
map them onto cores. It is the standard way to fan work out in orchestration
code.

Nesting is permissive: `pl.parallel` inside `pl.parallel`, `pl.range` inside
`pl.parallel`, `pl.parallel` inside `pl.range` all parse. There is no
"`pl.parallel` may not nest" rule.

`break` and `continue` are **not** allowed inside a `pl.parallel` loop (the IR
verifier rejects it); use `pl.range` if you need early exit.

## `pl.unroll` — compile-time replication

```python
for i in pl.unroll(4):
    t = pl.add(t, 1.0)
```

- Bounds must be **compile-time constants**. A runtime bound fails with:

  ```text
  ParserSyntaxError: pl.unroll() requires compile-time constant integer bounds
  ```

- `init_values=` is **rejected** — unrolled loops carry no loop state. Use
  `pl.range` when you need an accumulator.
- The step must be non-zero, and the trip count is capped (1024) by the unroll
  pass.

Use it for a short trip count whose body is small, so loop overhead and index
arithmetic disappear.

## `pl.pipeline` — software pipelining

```python
for kb in pl.pipeline(K // K_TILE, stage=2):
    tile_a = x[mb : mb + M_TILE, kb * K_TILE : (kb + 1) * K_TILE]
    tile_b = w[kb * K_TILE : (kb + 1) * K_TILE, :]
    acc = pl.matmul_acc(acc, tile_a, tile_b, init_cond=(kb == 0))
```

- `stage=` is a **required positive integer** — the pipeline depth, typically 2
  or 4. The loop body is replicated `stage` times so buffers ping-pong, the
  outer trip count advances by `stage * step`, and a tail dispatch covers the
  remainder.
- `init_values=` works exactly as in `pl.range`.
- With static bounds the step must be a constant; with dynamic bounds it must be
  positive.

This is the canonical inner loop of a tiled matmul: each iteration loads the
next K-slice of both operands and accumulates.

## Loop-carried state: `init_values` and `pl.yield_`

This is the one part of the loop syntax with sharp edges. The pattern is:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="sum"):
    acc = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
    for kb, (s,) in pl.range(HIDDEN // TILE, init_values=(acc,)):
        sq = pl.row_sum(pl.mul(x[r : r + ROW_TILE, kb * TILE : (kb + 1) * TILE],) ...)
        s = pl.add(s, sq)
        s_out = pl.yield_(s)
    y[r : r + ROW_TILE, :] = s_out
```

The rules:

1. **The loop target becomes a tuple** when `init_values` is present:
   `for i, (s,) in pl.range(N, init_values=(acc,))`. A single carried value is
   still written as a one-element tuple.
2. **The number of carried names must match** the number of `init_values`
   (`Mismatch: N iter_args but M init_values`).
3. **`pl.yield_` must be an assignment**, `s_out = pl.yield_(s)`. A bare
   `pl.yield_(s)` statement is rejected.
4. **The value you use after the loop is the yield's left-hand name**, not the
   name in the loop header. In the example above the result is `s_out`, and
   referencing `s` after the loop would be wrong (and, under-yielding is a parse
   error: `For loop has 2 iteration arguments but 1 return variables`).
5. Yields are matched **positionally**. Swapping the order of the left-hand
   names silently swaps the values.
6. **Over-yielding is a trap.** Yielding *more* values than there are carried
   arguments parses successfully and the extras are silently dropped; the error
   appears much later, from `SSAVerify`: `ForStmt YieldStmt value count (3) !=
   iter_args count (2)`. Count your yields.

## While loops

There are two spellings. The **natural** one is ordinary Python:

```python
i = 0
while i < 4:
    y[r : r + TILE, :] = pl.add(x[r : r + TILE, :], 1.0)
    i = i + 1
```

It produces a `WhileStmt` with no loop-carried arguments; the parser leaves the
phi nodes to the `ConvertToSSA` pass, and variables assigned in the body leak to
the enclosing scope (which is what makes the counter above work). The condition
must be a `Bool` scalar.

The **explicit** spelling is `pl.while_`, written as a `for` over an iterator
that yields once, with the condition declared by `pl.cond(...)` as the **first
statement of the body**:

```python
for (i,) in pl.while_(init_values=(0,)):
    pl.cond(i < N)
    i_next = i + 1
    i_out = pl.yield_(i_next)
```

- `init_values=` is **required** — a while loop must carry state.
- `pl.cond(...)` sets the loop condition; it is **not** a general branching
  helper, and despite the name it has nothing to do with `if`.
- The condition must be a `Bool`-typed scalar.

Neither while form is used anywhere in this repository — no example, no model
and no test uses a `while` loop, `pl.while_` or `pl.cond`. Device measurement
(september 2026 re-verification, Ascend 910B4, `-p a2a3`) backs the caution:

- The **natural** form carrying a tile across iterations is rejected by
  `ConvertToSSA`: `SSAVerify: Variable 'acc' used outside its defining scope`.
- The **explicit** `pl.while_` form parses and dispatches, but a
  three-trip loop over a carried tile and counter executed **zero or one
  trips** depending on where the store was placed — the carried condition did
  not re-evaluate as written.

Prefer `pl.range` with a computed trip count. Treat `pl.while_` as parseable
but not production-safe until a working kernel exists in this repository.

## `pl.spmd` — block-parallel dispatch

`pl.spmd(core_num)` dispatches `core_num` blocks in parallel. Only the block
count is positional; start is fixed at 0 and step at 1. It has **three forms**:

```python
# 1. Loop form: the body is auto-outlined to InCore; i is the block index
for i in pl.spmd(4):
    off = i * 16
    y[off : off + 16, :] = pl.add(x[off : off + 16, :], 1.0)

# 2. Context form: dispatch a pre-defined InCore kernel, or an inline body
with pl.spmd(4):
    y = sub_kernel(x, y)

# 3. Capture form: same as 2, plus the dispatch's TaskId for later deps
with pl.spmd(4, name_hint="stage1") as tid:
    y = sub_kernel(x, y)
```

- In the loop form the iteration variable **is** the block index, equivalent to
  `pl.tile.get_block_idx()`.
- A `pl.spmd` body must either read the block index or dispatch exactly one
  kernel; a body that does neither is rejected, because every block would run
  identical work.
- `with pl.spmd(...)` is *not* wrapped in a `pl.at`: the scope carries its own
  InCore region. `pl.at` and `pl.spmd` never nest inside each other in this
  repository.
- Keyword arguments: `sync_start=False`, `name_hint=""`,
  `optimizations=[...]`, `deps=[...]`, `allow_early_resolve=False`,
  `predicate=(...)`. A `pl.spmd` nested inside `pl.cluster()` loses the
  submit-only keywords (`deps=`, `allow_early_resolve=`, `predicate=`, `as tid`),
  because it is folded into a group function instead of becoming a task.

`pl.spmd` is the main fan-out mechanism in production kernels — it appears
hundreds of times in `models/`.

## `pl.split_aiv` — explicit AIV split

```python
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
    ...
```

`n` must be 2 and `mode=` is required. Related work-splitting helpers
`pl.aiv_shard` and `pl.aic_gather` inherit the split mode from the enclosing
region - and must **not** be given one. Writing `pl.aiv_shard(t, split=1)` inside
the loop is a parse error: `pl.aiv_shard() does not take a split= argument inside a
'for ... in pl.split_aiv(...)' loop — the split mode is inherited from that scope`.
They are also invalid outside such a region, and their operand must be an `Acc`
matmul result; see
[Tensor and system operations](07-tensor-and-system-operations.md). This is an expert construct (23 uses). The `AivSplitValid` verifier rejects
`pl.split_aiv` when it is written **directly inside a function whose type is
already InCore** (a `@pl.jit.incore` body) — author it in a plain `@pl.jit` /
`@pl.function` (Opaque) body and the outliner wraps the region in its own
InCore scope.

## Conditionals

`if` / `elif` / `else` are supported and become an IR `IfStmt`.

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="branch"):
    if n > 0:
        y[:, :] = pl.add(x[:, :], 1.0)
    else:
        y[:, :] = pl.sub(x[:, :], 1.0)
```

Rules and traps:

- **Conditions must be `Bool`-typed scalars.** There is no Python truthiness:
  `if my_int:` is rejected. Compare explicitly.
- **There is no static `if`.** A constant condition such as `if FLAG:` still
  emits an `IfStmt`; a later pass folds it. Do not use `if` to select between
  kernels at compile time.
- **Only one comparison per condition.** `0 < n < 10` is rejected
  (`Only simple comparisons supported`); write `n > 0 and n < 10`.
  `in`, `not in`, `is` and `is not` are rejected outright.
- **Values that escape a branch must be yielded:**

  ```python
  if n > 0:
      t2 = pl.yield_(pl.add(t, 1.0))
  else:
      t2 = pl.yield_(pl.sub(t, 1.0))
  y[:, :] = t2
  ```

  Without the yields, the `if` has no result and `t2` is not defined after it.

Because there is no ternary operator, the yield form above is how you write a
conditional value.

## `break` and `continue`

Both parse, with one restriction: **the innermost loop must be `pl.range` or a
while loop.** They are rejected inside `pl.parallel` and `pl.unroll` by the IR
verifier. Inside `pl.pipeline` the parser accepts them, but no pass or test
covers that combination — do not rely on it.

## Legality: what is enforced where

The parser enforces very little about *where* a loop may appear. Most placement
rules are checked later, by IR verifiers, or not at all.

| Construct | Enforced by | Restriction |
|---|---|---|
| `pl.unroll` | parser | constant bounds, non-zero step, no `init_values` |
| `pl.pipeline` | parser + verifier | required positive `stage=`; static step for static bounds |
| `pl.parallel` | verifier | no `break` / `continue` |
| `pl.split_aiv` | verifier | rejected when authored directly inside an InCore-typed function (a `@pl.jit.incore` body); write it in an Opaque body |
| `pl.spmd` | parser | body must use the block index or dispatch a kernel |
| `while cond:` | parser | condition must be a `Bool` scalar |
| `pl.while_` | parser | requires `init_values` and a leading `pl.cond` |
| loop nesting | nothing | all combinations parse |

A practical consequence: an InCore region normally uses `pl.range`,
`pl.pipeline` and `pl.unroll`; `pl.parallel` and `pl.spmd` are orchestration
constructs. The parser will not stop you from putting `pl.parallel` inside a
`pl.at` block, but the result is not something the compiler is designed for.

## Dynamic bounds

`pl.range`, `pl.parallel` and `pl.pipeline` accept runtime `pl.Scalar` bounds;
`pl.unroll` does not. `pl.spmd` accepts a single Scalar or a composite dynamic
expression as the block count. Keep tiling constants static — they shape the
generated code — and let only the outermost extent be dynamic. See
[Dynamic Shapes](08-dynamic-shapes.md).

## See also

- [Scopes and Dependencies](05-scopes-and-dependencies.md) — `pl.at`, `deps=`
  and `pl.submit`.
- [Dynamic Shapes](08-dynamic-shapes.md) — runtime bounds and dimensions.
- [Tile Operations](06-tile-operations.md) — what goes inside the loops.