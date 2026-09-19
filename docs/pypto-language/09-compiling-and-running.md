# Compiling and Running

This page is the API reference for turning a written kernel into something that
runs: the decorators, the compile/run entry points, platforms, caches, and the
debugging surfaces that show you what the compiler produced.

## The decorators as API

```text
@pl.jit(func=None, *, auto_scope=True) -> JITFunction
```

| Property | Behaviour |
|---|---|
| Parse time | **first call** (or `.lower()` / `.compile()`), not at decoration |
| Default specialization | `FunctionType.Orchestration`, `Level.CHIP`, `Role.Orchestrator` |
| `auto_scope` | `True` lets the compiler place runtime scopes; `False` means you place them |

```text
@pl.function(func=None, *, type=pl.FunctionType.Opaque, level=None, role=None,
             attrs=None, auto_scope=True, strict_ssa=False, external_source=None)
    -> ir.Function
```

| Property | Behaviour |
|---|---|
| Parse time | **decoration time** — i.e. module import |
| Returns | `ir.Function` immediately |
| `attrs` | deprecated; it warns and points at `pl.func_attr` |

`@pl.inline(func)` returns an `InlineFunction`, and
`@pl.program(cls=None, *, strict_ssa=False)` returns an `ir.Program` at
decoration time.

The `@pl.jit.<kind>` sub-decorators are described in
[Programming Model](01-programming-model.md#the-decorator-tiers). Two API notes:
only `@pl.jit.incore` accepts `level=`, and only `@pl.jit.host` and
`@pl.jit.inline` accept `auto_scope=`. Passing either to a kind that does not
support it raises `TypeError`.

## `JITFunction` methods

| Method | Purpose |
|---|---|
| `fn(*tensors, scalars...)` | compile if needed, then execute on device; returns the outputs |
| `fn.compile(*samples, config=...)` | build and return a `CompiledProgram` without running |
| `fn.warmup(*samples)` | compile without running |
| `fn.specialize(*samples)` | run the front-end and return the specialized `ir.Program` |
| `fn.lower(*samples)` | lower to IR; handy for inspecting the outlined kernels |
| `fn.param_names` | the kernel's parameter names |
| `fn.output_param_names` | the names of its `pl.Out` **and `pl.InOut`** parameters |

### Two call conventions

This trips people up, so it is worth stating plainly:

```python
# JITFunction.__call__ — every parameter, INCLUDING the pl.Out tensors
y = kernel(x, y, a)

# CompiledProgram — return style; omit the Out tensors and receive them back
prog = kernel.compile(x, y, a)
y = prog(x, a)
```

Calling a `JITFunction` without an `pl.Out` argument fails with a missing
argument error; the return-style convenience exists only on the compiled
program.

## What happens on a call

1. Capture the enclosing namespaces and refresh dependencies.
2. Classify arguments into tensors and scalars.
3. Build a cache key from tensor metadata and referenced constants (dynamic
   dimensions become `None` in the shape tuple).
4. **Cache hit** — run the cached artifact and return.
5. **Cache miss** — specialize the entry plus its dependencies, `pl.parse`,
   `ir.compile`, store the artifact, then execute.

Because the cache key includes shapes, a static-shape kernel recompiles for each
new shape; a symbolic-dimension kernel does not. This is the practical reason to
use dynamic shapes.

## Platforms

| Value | Meaning |
|---|---|
| `a2a3` | real A2/A3 device |
| `a2a3sim` | A2/A3 simulator |
| `a5` | real A5 device |
| `a5sim` | A5 simulator |

Two constraints worth knowing:

- **One architecture per process.** Switching architectures in-process aborts
  with `Backend type already set to Ascend910B, cannot change to Ascend950`,
  because the backend selection is set once. Run a2a3 and a5 in separate
  processes.
- **Simulator runs need GCC 15 on `PATH`.** The lookup
  (`runtime/simpler_setup/toolchain.py`) wants the **version**, primarily under
  the bare name `g++-15`; a prefixed variant such as
  `aarch64-linux-gnu-g++-15` or an absolute path is also accepted, but a
  toolchain reachable only as `g++` is not. The probe's failure hint is
  `g++-15 not found. Please install g++-15.` Provide GCC 15 under the names
  `g++-15` and `gcc-15` — how you install or alias it is an environment
  concern, and **not** something this repository ships or prescribes. Confirm
  it resolves before running:

  ```bash
  conda activate pypto
  g++-15 --version
  python examples/beginner/hello_world.py -p a2a3sim
  ```

  Real-device runs (`-p a2a3 -d 0`) do not need it.

## `RunConfig`

`RunConfig` is keyword-only, and `platform=` is a **constructor-only** spelling
that maps to an architecture plus an execution mode. Pass it through the
harness's `config=dict(...)` rather than constructing it yourself.

Fields you will actually use:

| Field | Meaning |
|---|---|
| `platform` | `a2a3` / `a2a3sim` / `a5` / `a5sim` |
| `device_id` | which device to run on |
| `enable_chip_swimlane` | profiling swimlane capture level |
| dump options | `enable_dump_args` and friends; `0` off, `1` partial, `2` hybrid (metadata for every task, payload only for marked args), `3` full (everything, heaviest) |
| `distributed_config` | `DistributedConfig(device_ids=[...])` for multi-chip |

A trap with `enable_chip_swimlane`: in `RunConfig`, `True` means level **4**,
but the repository's own CLI pattern `nargs="?"` with `const=1` makes a bare
`--enable-chip-swimlane` mean level **1**. Always pass an explicit integer.

`RunConfig.codegen_only` exists in the type but is **never read** — it is a dead
field.

## Caches and build directories

- The generated code lands in a build directory that is **relative to the
  current working directory**: `build_output/` unless `PYPTO_PROG_BUILD_DIR`
  overrides it. Run kernels from a consistent directory or you will rebuild.
- There is a persistent artifact cache (`~/.cache/pypto/jit` by default, keyed by
  an artifact manifest). It is **disabled by default**.
- **Leave it disabled in this workspace.** Enabling it
  (`PYPTO_CACHE=1` or `CacheConfig(enabled=True)`) was measured to *recompile on
  every call* here, because the local `ptoas` launcher's
  `#!/usr/bin/env python3` shebang is rejected by the toolchain probe, which
  silently marks the toolchain unusable and bypasses the cache. The only signal
  is a logger line at `info` level. If you do enable it, watch the cache stats
  for bypasses.
- Clearing the cache means deleting its root directory.

Environment variables you may need: `PYPTO_PROG_BUILD_DIR`, `PYPTO_CACHE`,
`PYPTO_CACHE_DIR`, `PYPTO_CACHE_READONLY`, `PYPTO_COMPILE_PROFILING`,
`PYPTO_LOG_LEVEL` (C++ side only), `PYPTO_RUNTIME_LOG`, `PYPTO_EMIT_PTO_LOC`,
`PYPTO_VERIFY_LEVEL`, `PYPTO_WARNING_LEVEL`, `PYPTO_REBUILD_FROM_PTO`.

### Stale artifacts

Editing generated C++ in a build directory is **not** picked up by a plain call,
because the work directory carries its own cache. Use the replay driver, which
invalidates before running:

```bash
python -m pypto.runtime.debug.replay <work_dir>
```

## Seeing what the compiler produced

| Tool | Use |
|---|---|
| `fn.lower()` | print the IR, including the outlined `k_incore_*` kernels |
| `dump_passes=True` | write one file per pass stage; a full run produces 55 files (frontend plus 54 passes) |
| `python -m pypto.tools.ir_trace` | render an HTML trace of the IR across passes |
| `pypto.debug.torch_codegen` | emit a PyTorch reference implementation of the post-pass IR |
| `pl.static_print(...)` | log compile-time values while parsing |
| `pl.static_assert(cond, msg)` | fail the build with your own message |

Two names show up **only** in that output and in signatures, never in code you
write - they are the last two `pl.*` exports that no kernel calls:

- `pl.MemRefType` - the annotation the printer emits for a bare `MemRef`-typed
  variable. Seeing `pl.MemRefType` in `fn.lower()` output is expected; you do not
  annotate with it yourself.
- `pl.IntLike` - the union alias (`int | pl.Scalar[...]`) behind parameters such
  as `event_id`. It documents what a signature accepts; it is not a value.

The pass list has **54** entries. The developer documentation lists 52 and omits
`NormalizeStmtStructure`, so every index past the fifth is off by one in those
docs; trust the emitted dump, not the table.

## Validating a kernel

Kernels in this repository ship with a golden harness: a spec list, a PyTorch
reference, and a `run(...)` call.

```python
def build_tensor_specs(batch: int = BATCH, hidden: int = HIDDEN):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [batch, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("y", [batch, hidden], torch.float32),
    ]


def golden_rms_norm(tensors):
    import torch

    x = tensors["x"]
    gamma = tensors["gamma"]
    tensors["y"][:] = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * gamma
```

Rules the harness enforces:

- **`golden_fn` takes exactly one argument** (a dict of tensors) and must write
  its outputs **in place**. That is why every reference ends in
  `tensors["y"][:] = ...` and never returns anything.
- `TensorSpec(name, shape, dtype, init_value=...)` describes an input; an output
  needs no `init_value`. Direction comes from the kernel's annotation, not from
  the spec, and reading `is_output` before the harness stamps it raises.
- `TensorSpec(..., resident=...)` keeps a parameter resident on device
  (`child_memory`): uploaded once and reused across the validation dispatch and
  every benchmark round, skipping the per-dispatch copies. **L3 only.** Values:
  `None` / `False` (off, the default), an `int` worker id for whole-tensor on
  the card whose kernel consumes it, or `"stacked"` to shard a
  `[world_size, *tail]` leading dim per rank. `True` is rejected as ambiguous.
- `ScalarSpec(name, dtype, value)` describes a scalar argument.
- `run(fn=..., specs=..., golden_fn=..., config=dict(platform=..., device_id=...), rtol=..., atol=...)`
  returns a result with `.passed`; the conventional failure tail is:

  ```python
  if not result.passed:
      if result.error:
          print(result.error)
      raise SystemExit(1)
  ```

Import `golden` by running from the repository root with `PYTHONPATH="$PWD"`.
It currently also works through an inherited `PYTHONPATH` that happens to end in
a colon (the empty entry means the current directory), which is fragile — set it
explicitly.

### Comparators

`run(...)` accepts different comparison functions, and the default is **not
strict**:

| Comparator | Notes |
|---|---|
| `ratio_allclose` | default `max_error_ratio=0.005`, i.e. 0.5% of elements may be outliers |
| `ratio_reldiff` | relative-difference variant; unlike `ratio_allclose` it has no `ignore_nan` |
| `mapped_pool_ratio_allclose` | validates an index mapping; rows outside the mapping must match exactly |
| `mapped_pool_ratio_reldiff` | the same slot-mapped paged-pool validation, but scoring mapped rows with `ratio_reldiff` instead of `ratio_allclose` |
| `topk_pair_compare` | for top-k: checks the *values* stay monotone across mismatched index positions |
| `error_distribution` | a measurement, not a gate — it always passes |

When chasing a precision bug, start from `ratio_allclose(max_error_ratio=0.0)`,
use `error_distribution` to see the shape of the error, then relax to a
documented ratio.

A caution about the test suite: `tests/golden` can pass against a **stub**
implementation of pypto substituted when the real package is absent, and CI does
not install pypto. A green harness run is not by itself evidence that the
compiler works — run a real kernel.

## Repository conventions for a kernel file

Every kernel file in this repository follows the same skeleton:

1. the 8-line license header;
2. a module docstring stating the formula and the tiling;
3. `import pypto.language as pl`;
4. `ALL_CAPS` tiling constants, each commented with what it is sized by;
5. the `@pl.jit` entry function, ending in `return y`;
6. `build_tensor_specs(...)`;
7. `golden_<entry_fn_name>(tensors)`;
8. an `if __name__ == "__main__":` block with `-p`, `-d` and
   `--enable-chip-swimlane` flags and the `run(...)` call.

```bash
# simulator (requires g++-15 on PATH — see the toolchain note above)
python examples/beginner/hello_world.py -p a2a3sim

# real device
python examples/beginner/hello_world.py -p a2a3 -d 0
```

## Lint rules that touch documentation and code

| Check | Scope |
|---|---|
| `python tests/lint/check_headers.py` | `.py` files only; exact 8-line header equality |
| `python tests/lint/check_docs_nav.py` | every page under `docs/` must appear in the mkdocs `nav` exactly once |
| `ruff check .` | `select = ["F"]` only — pyflakes. Line length is configured but **not** enforced, and `# noqa: PLR...` comments in existing kernels are inert |

`ruff format` is not wired into pre-commit; running it would rewrite most of the
repository.

## See also

- [Programming Model](01-programming-model.md) — decorator tiers.
- [Dynamic Shapes](08-dynamic-shapes.md) — why symbolic dims avoid recompiles.
- [Distributed Kernels](10-distributed.md) — `DistributedConfig` and launch.