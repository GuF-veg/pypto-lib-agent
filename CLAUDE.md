# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PyPTO-Lib** is a primitive tensor function library built on the pypto programming framework. It defines tensor-level operations (analogous to PyTorch ATen) that the compiler tiles and lowers to PTO-ISA instructions. The library does not fix incore/orchestration boundaries — that is decided by the backend and runtime.

## Architecture

### Stack

```
User Python Code (pypto frontend)
    ↓
pypto Compiler (IR, passes, codegen)
    ↓
PTO-ISA (.pto MLIR) → ptoas toolchain → device kernels
    ↓
PTO2 Runtime (simpler) → Ascend AICPU/AICore
```

### Tensor vs Tile

- **Tensor** (`pl.Tensor[[M, N], pl.FP32]`): N-D logical tensor in global memory (DDR)
- **Tile** (`pl.Tile[[16, 16], pl.FP16]`): Block of data in on-chip buffer (UB/L1/L0A/L0B)
- **No type conversion between them**: data movement uses `pl.load` (Tensor→Tile) and `pl.store` (Tile→Tensor)

### Three Function Types

| Type | Where | Purpose |
|------|-------|---------|
| `InCore` | AICore | Load/compute/store onchip — explicit memory placement |
| `Orchestration` | AICPU | Build task graph, call InCore kernels, allocate tensors |
| `Opaque` | Compiler decides | Use `with pl.at()` for anonymous InCore scopes |

### InCore Scope

`with pl.at(level=pl.Level.CORE_GROUP):` marks an anonymous InCore region. The compiler derives arguments: inputs (outside, read-only), inouts (outside, modified), outputs (defined outside, written in scope, read after). Scope is outlined into a named function + call site.

### Memory Spaces

| Space | Alias | Purpose |
|-------|-------|---------|
| `pl.MemorySpace.Vec` | UB | Vector unit buffer |
| `pl.MemorySpace.Mat` | L1 | Matrix buffer (matmul staging) |
| `pl.MemorySpace.Left/Right` | L0A/L0B | Matmul operands |
| `pl.MemorySpace.Acc` | Acc | Accumulator |

Matmul pipeline: `load(mat) → move(left/right) → matmul → store`

## Directory Structure

```
examples/
  beginner/      hello_world, matmul
  intermediate/  softmax, rms_norm, layer_norm, rope
  models/        Qwen3-32B, DeepSeek V3.2
docs/            pypto-frontend-coding-style, pto2_rt, para_for
tests/
  golden/        Golden harness (pytest-based)
  lint/          check_headers.py, check_english_only.py
build_output/    Generated codegen artifacts (gitignored)
```

## External Dependencies

| Dependency | Repository | Purpose |
|------------|-----------|---------|
| pypto | `hw-native-sys/pypto` | Compiler framework (IR, codegen, passes) |
| ptoas | `hw-native-sys/PTOAS` | PTO assembler & optimizer toolchain |
| simpler | `hw-native-sys/simpler` (pypto submodule) | Runtime (pto-rt2) |

## Environment Setup

Use `/setup_env` skill or see `.claude/skills/setup_env/SKILL.md` for manual steps. Required env vars:

| Variable | Purpose |
|----------|---------|
| `PTOAS_ROOT` | Path to ptoas binary |
| `PTO_ISA_ROOT` | Path to pto-isa |
| `ASCEND_HOME_PATH` | CANN install path (device only) |

## Common Commands

### Running Examples

```bash
# Single example (simulator)
python examples/beginner/hello_world.py

# Specify platform
python examples/beginner/hello_world.py -p a2a3sim    # x86 simulation
python examples/beginner/hello_world.py -p a5sim      # ARM simulation
python examples/beginner/hello_world.py -p a2a3 -d 0 # real device

# Run all beginner + intermediate
for f in $(find examples/beginner examples/intermediate -name '*.py' ! -name '*draft*' | sort); do
  python "$f" -p a2a3sim
done
```

### Testing

```bash
# Unit tests (golden harness)
pytest tests/golden -v

# Run a single test file
pytest tests/golden/test_runner.py -v

# Lint checks (pre-commit style)
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
```

### Checking Output

```bash
# View generated codegen
ls build_output/
# Subdirs: orchestration/*.cpp, ptoas/*.pto, ptoas/*.cpp, kernels/aiv/*.cpp
```

### Pre-commit

```bash
pip install pre-commit
pre-commit run --all-files
```

## Key Documentation

| File | What It Covers |
|------|---------------|
| `README.md` | Library architecture, tensor vs tile, tiling, fusion |
| `docs/pypto-frontend-coding-style.md` | Python frontend syntax, types, InCore/Orchestration/Opaque, PTO codegen |
| `docs/pto2_rt.md` | PTO2 runtime design: task ring, heap ring, TensorMap, scheduler |
| `docs/para_for.md` | Unified loop grammar: `pl.range` + `parallel`/`chunk` kwargs |
| `.claude/rules/coding-style.md` | Naming conventions, file structure, GM alignment |
| `.claude/rules/known-issues-tracking.md` | Log bugs to `KNOWN_ISSUES.md` |

## Coding Conventions

1. **Import**: `import pypto.language as pl` (never `ir` or other aliases)
2. **Types**: `pl.Tensor`, `pl.Tile`, `pl.Scalar` with explicit shapes and dtypes
3. **Parameters**: `pl.Out[type]` / `pl.InOut[type]` for direction; scalars cannot be InOut
4. **GM alignment**: All `pl.slice` of global memory must be >= 512 bytes
5. **Constants**: UPPER_SNAKE_CASE at module level (e.g., `HEAD_DIM = 128`)
6. **Naming**: `PascalCaseProgram`, `snake_case_function`, `DESCRIPTIVE_TENSOR` variables
7. **File header**: Copyright → docstring → imports → constants → `@pl.program` class → `if __name__ == "__main__":`
8. **All code comments and documentation in English**

## GM Tensor Alignment Rule

All `pl.slice` of global memory tensors must be >= 512 bytes. Violating this causes hardware faults. Check tensor shapes and offsets before slicing.

## Known Issues Tracking

Log encountered bugs/defects to `KNOWN_ISSUES.md` (gitignored, local-only). Format: brief title, date, found-during context, description, location, severity. Remove entries when resolved. Use `/create-issue` to promote to GitHub.
