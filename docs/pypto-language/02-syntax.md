# Syntax

PyPTO functions are ordinary Python functions that a parser reads **at compile
time**. The parser does not execute your function; it walks the abstract syntax
tree. So the question "is this legal PyPTO?" always means "does the parser
accept this AST node?" — and the answer is a closed allowlist, not "anything
Python can do".

## The parser needs your source

Because the parser reads the AST rather than running the code:

- **Kernels must live in a real file.** A function defined in a REPL, in
  `exec(...)`, or in a string that `compile()` built has no retrievable source
  and fails with:

  ```text
  ParserSyntaxError: Cannot retrieve source code for function 'f'
  ```

- **`@pl.function` fails at import time**, `@pl.jit` fails at first call (or at
  `.compile()`). A syntax error in a `@pl.jit` kernel therefore surfaces when
  you run it, not when you import the module.
- Everything the parser needs must be *statically visible*: a helper function
  in the same file is fine to reference as a constant source of values, but you
  cannot **call** an arbitrary Python function from inside a kernel body (see
  [Calls](#calls)).

## Statements

| Python statement | Accepted? | Notes |
|---|---|---|
| `x = expr` | yes | single target, or a tuple target for multi-output ops |
| `x: T = expr` | yes | annotation optional; target must be a plain name |
| `x, y = expr` | yes | unpacks multi-output ops and `pl.submit` results |
| `for ... in pl.<loop>(...)` | yes | iterator must be a PyPTO loop helper |
| `while cond:` | yes | natural form; becomes a `WhileStmt` (non-SSA; a later pass converts it) |
| `for ... in pl.while_(...)` | yes | the explicit loop-carried form; see [Control Flow](04-control-flow.md#while-loops) |
| `if` / `elif` / `else` | yes | becomes an `IfStmt`; see [Conditionals](04-control-flow.md#conditionals) |
| `with pl.at(...)` and friends | yes | exactly one context manager, from a fixed set |
| `return a` / `return a, b` | yes | optional; `-> None` is an error |
| bare call, e.g. `pl.static_print(...)` | yes | expression statement |
| `pass` | yes | |
| `break` / `continue` | yes | only when the innermost loop is `pl.range` or a while loop |
| `x += 1`, `x -= 1`, ... | **no** | `UnsupportedFeatureError: Augmented assignment is not supported in DSL functions: t += 1.0` |
| `a = b = c` (chained) | **no** | |
| `a, *rest = ...` | **no** | starred targets rejected |
| `def` inside a body | **no** | `UnsupportedFeatureError: Unsupported statement type: FunctionDef` |
| `import` inside a body | **no** | `UnsupportedFeatureError: Unsupported statement type: Import` |
| `try` / `except` | **no** | |
| `match` | **no** | |
| `assert` | **no** | use `pl.static_assert(...)` for compile-time checks |
| `del`, `global`, `nonlocal`, `raise` | **no** | |
| `class` inside a body | **no** | |

Write `x = x + 1`, not `x += 1`. This is the single most common syntax mistake
when moving Python code into a kernel.

## Expressions

The parser accepts a fixed set of expression nodes:

| Expression | Accepted? |
|---|---|
| names, integer/float/bool/string constants | yes |
| `a + b`, `a - b`, `a * b`, `a / b`, `a // b`, `a % b`, `**` | yes |
| one comparison: `a < b`, `a <= b`, `a == b`, `a != b`, `a > b`, `a >= b` | yes |
| `a and b`, `a or b` | yes |
| `not a`, `-a`, `+a` | yes |
| calls (`pl.add(...)`, `pl.at(...)`, a sub-kernel call) | yes, restricted (see below) |
| attribute access (`pl.Level.CORE_GROUP`, `self.kernel`) | yes |
| subscript / slice (`x[i, j]`, `x[a:b, c:d]`, `arr[i]`) | yes |
| list literal `[a, b]`, tuple literal `(a, b)` | yes |
| chained comparison `0 < n < 10` | **no** — `Only simple comparisons supported` |
| `in` / `not in` / `is` / `is not` | **no** — `Unsupported comparison: In` / `Is` |
| ternary `a if cond else b` | **no** — `UnsupportedFeatureError: Unsupported expression type: IfExp` |
| lambda | **no** — `Unsupported expression type: Lambda` |
| dict / set literal | **no** |
| comprehension (`[... for ...]`) | **no** — the one exception is `deps=[...]` on `pl.submit` |
| f-string | **only as a `pl.static_print` argument**; as a general expression it is `Unsupported expression type: JoinedStr`, and conversions/format specs (`!r`, `:.2f`) are rejected even in `static_print` |
| walrus `:=` | **no** |
| starred `*args` / `**kwargs` in a call | **no** |

Because there is no ternary, express a conditional value with an `if` that
yields, and because there is no comprehension, spell loops out:

```python
# instead of: v = 1.0 if n > 0 else 2.0
with pl.at(level=pl.Level.CORE_GROUP, name_hint="pick"):
    if n > 0:
        v = pl.yield_(1.0)
    else:
        v = pl.yield_(2.0)
```

### Calls

Three kinds of call are legal:

1. **PyPTO operations** — `pl.add(...)`, `pl.load(...)`, `pl.matmul(...)`. These
   are resolved against the operator table, so an unknown keyword is a hard
   error:

   ```text
   InvalidOperationError: pl operation 'add': add() got an unexpected keyword
   argument 'nonexistent_kwarg'
   ```

2. **PyPTO constructs** — `pl.parallel(...)`, `pl.at(...)`, `pl.range(...)`.
   These are recognised syntactically; an unknown keyword gives
   `Unknown keyword argument '<k>' in pl.range()`.

3. **Other kernel functions** — calling another `@pl.function`/`@pl.jit.incore`
   kernel from an orchestration body. This becomes an IR `Call`, i.e. a task
   launch; it is **never inlined**. Inlining happens only for functions marked
   `@pl.inline` / `@pl.jit.inline`.

Anything else — a plain Python helper function — is rejected:

```text
UnsupportedFeatureError: Unsupported function call: helper(2)
```

Python helpers are still useful *outside* the kernel: they run at parse time
when their result is a constant. `TILE = compute_tile_size()` at module level
works; `pl.add(x, compute_tile_size())` inside a body does not.

## Annotations

- **Parameter annotations are mandatory.** Every parameter needs a type.
- **Local variable annotations are optional** — most kernels write
  `t = pl.load(...)` with no annotation, and real code does so far more often
  than not.
- When a local annotation *is* present it is checked against the inferred type
  (kind, dtype, rank, static dimensions). See
  [Types and Annotations](03-types.md).
- The return annotation is optional, and `-> None` is rejected. Write
  `return y` with no arrow, or annotate the real type.

## Reassignment and SSA

The default mode is **not** strict SSA: you may reassign a name, which is how
accumulators are written.

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="acc"):
    t = x[r : r + TILE, :]
    t = pl.add(t, 1.0)
    t = pl.mul(t, 2.0)
    y[r : r + TILE, :] = t
```

The one rule is **type stability**: reassigning a name to a different type is an
error (`Cannot reassign ... different type`). An unset `Tile` memory space is
treated as a wildcard and may be refined once.

`strict_ssa=True` is available on `@pl.program` (and is silently ignored when
passed to an inner `@pl.function` inside a program). Under strict SSA,
subscript writes such as `y[...] = ...` are illegal, which is why kernels are
not written that way.

## There is no static `if`

A condition that is a Python constant still produces a runtime `IfStmt`:

```python
if FLAG:          # FLAG = True at module level
    y[:, :] = pl.add(x, 1.0)
else:
    y[:, :] = pl.sub(x, 1.0)
```

This *parses*, and the branch is folded later by a compiler pass — but do not
expect the parser to drop the dead branch, and do not use `if` to choose
between two kernels at compile time. Conditions must be `Bool`-typed scalars;
there is no Python truthiness, so `if my_int:` is not allowed.

## Compile-time constants and evaluation

Values that the parser can evaluate at parse time are folded into the IR, so
they cost nothing at run time:

- module-level constants (`ROW_TILE = 128`), including arithmetic on them;
- `pl.constexpr` parameters (**`@pl.jit` only**) — see
  [Types and Annotations](03-types.md#compile-time-constants);
- `pl.const(value, dtype)` for a typed constant;
- loop bounds and shapes built from the above.

Anything that depends on runtime data must be a `pl.Scalar[...]` parameter and
is evaluated on device.

## Comments

Comments in a kernel body are **preserved** and attached to the following
statement as leading comments, so they survive into the IR and can be seen in
IR dumps. One exception: a comment at the very end of a block, with no
statement after it, is dropped with a `UserWarning`. Keep explanatory comments
immediately above the statement they describe.

## Diagnostics

Error messages are Python exceptions with descriptive text. There is an
`error_codes.py` module defining codes such as `E001`, but **nothing raises
them** and no message carries a code — do not go looking for a code reference.
The useful classes are:

| Exception | Raised for |
|---|---|
| `ParserSyntaxError` | malformed or unsupported syntax, bad loop bounds, source not retrievable |
| `UnsupportedFeatureError` | an AST node outside the allowlist (statements, expressions, calls) |
| `InvalidOperationError` | a PyPTO op called with bad arguments or operand types |
| `TypeError` / `ValueError` | decorator misuse and construct-argument validation |

## `pl.parse` is not a text format

`pl.parse(source)` / `pl.loads(...)` look like a textual IR parser, but they
`compile()` and `exec()` **Python** source with `pl` and `pld` injected into the
namespace. Names that do not resolve become `pl.dynamic` dimension variables.
It is used by the `@pl.jit` path and by IR print-then-reparse round-trip tests —
not something you need when writing kernels.

The pair has older spellings, `pl.parse_program(...)` and
`pl.loads_program(...)`, which are **deprecated in their docstrings only** —
calling them emits no runtime warning; the docstring is the only pointer to
`pl.parse` / `pl.loads`. Older kernels in this repository still call them;
there is no reason to write new code against them.

Two restrictions on the text front end are worth knowing. First, the source it
parses must define its function with `@pl.function` or its class with
`@pl.program` — `@pl.jit` inside a text block is not recognized. Second, the
text sees only the names injected into its namespace (`pl`, `pld`) plus what
you pass in; module-level globals of the calling script are **not** visible, so
shapes must be literals or explicitly supplied.

## See also

- [Types and Annotations](03-types.md) — what goes in the annotations.
- [Control Flow](04-control-flow.md) — the loop and conditional forms.
- [Patterns and Pitfalls](11-patterns-and-pitfalls.md) — the mistakes that waste the most time.