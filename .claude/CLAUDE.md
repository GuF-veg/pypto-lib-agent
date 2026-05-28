# PyPTO-Lib Developer Guidelines

## Project Overview

PyPTO-Lib hosts tensor-level kernels and end-to-end LLM model
implementations built on the **pypto** programming framework, targeting
Ascend NPUs (910B/C, 950). It also ships a golden-validation test harness
(`golden/`).

## Repository Layout

- `examples/{beginner,intermediate,advanced}/` — self-contained kernels for learning the DSL
- `models/{qwen3,deepseek}/` — end-to-end LLM kernels by family
- `golden/` — test harness: compile, run on device, validate against torch
- `tests/` — lint checks and golden-fn unit tests
- `docs/` — coding-style and workflow reference
- `build_output/` — generated compilation artifacts (gitignored)

Files ending in `_draft.py` are works-in-progress and excluded from CI.

## Key Documentation

- `README.md` — project intro, quick start, dependencies
- `docs/pypto-coding-style.md` — **canonical** coding style: `pl.at` scopes, four loop constructs (`pl.range`/`pl.parallel`/`pl.pipeline`/`pl.spmd`), vector / cube / mte ops
- `docs/compile-runtime-workflow.md` — what `python <kernel>.py -p <platform>` does end-to-end (compile passes/codegen → runtime → golden → validate)

## External Dependencies

| Repo | Role |
|------|------|
| **pypto** | Tile-based programming framework — multi-level IR + codegen |
| **simpler** | PTO runtime — task graph build/execute on AICPU + AICore (submodule of pypto) |
| **ptoas** | LLVM/MLIR PTO Bytecode assembler/optimizer |
| **pto-isa** | PTO Tile Library — virtual tile-ISA implementations |

Pinned versions live in [.github/workflows/ci.yml](../.github/workflows/ci.yml).

## Environment Setup

Use the `/setup_env` skill, or refer to `.claude/skills/setup_env/SKILL.md`.

## Common Commands

```bash
# Run an example on the simulator
python examples/beginner/hello_world.py -p a2a3sim

# Run a model on real NPU device 0
python models/qwen3/14b/qwen3_14b_decode.py -p a2a3 -d 0

# Lint + header/English-only checks (mirrors the pre-commit CI job)
pre-commit run --all-files

# Golden harness unit tests (the only pytest suite; runs without a device)
pytest tests/golden -v
pytest tests/golden/test_runner.py::<test_name> -v   # single test
```

Every kernel script accepts `-p {a2a3, a2a3sim, a5, a5sim}` and `-d <device_id>`,
exits non-zero on validation mismatch, and writes artifacts under
`build_output/` (gitignored). CI (`.github/workflows/ci.yml`) auto-selects only
the changed `examples/` / `models/` files with a `__main__` for the `sim` and
`a2a3` matrix; the full set runs on push to `main`.

## Golden Harness Architecture

`golden/` is the single entry point every kernel uses to compile, execute, and
validate. The flow inside `golden.run(...)` is:

1. **Compile** the `@pl.program` for the target platform via pypto.
2. **Materialize inputs** from caller-provided `TensorSpec` / `ScalarSpec`
   (random or fixture-loaded torch tensors).
3. **Execute** the compiled kernel through the simpler runtime (sim or NPU).
4. **Compute the torch reference** with the user-supplied `golden_fn`.
5. **Validate** outputs via `validation.validate_golden` (per-spec tolerance,
   shape, dtype) and return a `RunResult`.

When debugging a failing kernel, the four stages above are the natural seams —
see `docs/compile-runtime-workflow.md` for the full pass list.

## Important Rules

1. **Read `docs/pypto-coding-style.md` first** before writing or modifying any kernel — it is the authoritative coding-style reference.
2. **`docs/compile-runtime-workflow.md`** explains the harness flow; consult it when debugging compile/runtime/validation failures.
3. **Consult `.claude/skills/`** for task-specific workflows (e.g. `setup_env/`, `bisect-precision/`).
4. **No private information** (usernames, absolute paths with usernames, etc.) in code or docs.
5. **All code comments and documentation in English** unless the user explicitly requests otherwise.
