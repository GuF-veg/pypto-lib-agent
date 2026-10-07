# PyPTO-Lib Codex Instructions

This repository keeps AI project policy and execution workflows in
`.claude/`. Public technical guidance is canonical in `docs/`; skills should
reference it rather than maintain a second copy. This file is only the Codex
entrypoint.

## Read First

Before making code changes, reviewing code, committing changes, or working on
kernel behavior:

- Read `.claude/CLAUDE.md`
- Read task-relevant files in `.claude/rules/` when present
- Follow `.claude/skills/*/SKILL.md` when the task matches a documented workflow
- Read `docs/pypto-coding/l2-programming.md` (with its `operations.md`,
  `loops.md`, and `naming-and-comments.md` siblings) before writing or
  modifying kernels

Task mapping:

- Operator development (write, verify, tune): `.claude/skills/operator-dev/SKILL.md`
- Environment setup: `.claude/skills/setup-env/SKILL.md`
- Kernel style pass: `.claude/skills/fmt-coding-style/SKILL.md`
- Precision debugging: `.claude/skills/bisect-precision/SKILL.md`
- Performance profiling: `.claude/skills/incore-profiling/SKILL.md`
- Profile feedback (pfdb): `.claude/skills/profile-feedback/SKILL.md`
- Cube tile tuning: `.claude/skills/cube-tile-tuning/SKILL.md`
- Commit workflow: `.claude/skills/git-commit/SKILL.md`
- PR workflow: `.claude/skills/github-pr/SKILL.md`
- PR review fixes: `.claude/skills/fix-pr/SKILL.md`
- Issue creation: `.claude/skills/create-issue/SKILL.md`

When a Claude skill or agent refers to `Task`, a subagent, or Claude-only
plugins:

- Execute the workflow directly in Codex
- Use parallel tool calls when safe
- Treat any agent-specific instructions as checklists, not as a separate runtime

## Working Agreements

- Keep changes scoped to the requested kernel, model, test, or documentation area
- Prefer existing project patterns and examples over new abstractions
- Keep public documentation and examples aligned when behavior changes
- Keep durable technical guidance in `docs/`; keep skills focused on
  environment-aware execution, safety, and reporting
- Do not commit generated build artifacts from `build_output/`
- Treat credentials, local paths with usernames, and machine-specific state as
  off-limits unless the user explicitly asks for them

## Runtime Environment

This project's runtime environment is the conda virtual environment `pypto`.
All commands must run within it:

```bash
conda activate pypto
```

Every `python`, `pytest`, `ruff`, or other tool invocation in this repository
must be executed inside this environment.

### Environment traps that cost real debugging time

- **Activate the environment for child tools too.** Use `conda activate pypto`
  before `npu-run`, or `conda run -n pypto npu-run python <kernel>.py -p a2a3 -d 0`.
  An absolute interpreter path alone does not activate `PATH` for the PTOAS
  launcher; see [PTOAS setup](docs/get-started/installation.md#ptoas).
- **Retry opaque runtime failures in a fresh process and retain the original
  diagnostic.** A multi-kernel sweep reported `prepare_native_run failed with
  code 13` for cases that behaved differently alone. Its trigger remains
  unverified; a retry helps distinguish a process-state problem from a kernel
  failure, but does not establish the cause.

## NPU Usage (Shared Device Pool)

The 8 Ascend NPUs on this machine are shared by multiple concurrent agents and
scheduled by `npu-run` (installed at `~/.local/bin/npu-run`, already in
`PATH`). Two processes using the same NPU at the same time fail, so:

- Wrap **every command that touches a real NPU** with `npu-run`: model and
  example runs on real hardware (`-p a2a3`/`-p a5`), `pytest` on the golden
  harness, benchmarks, and profiling tools alike. Never run them bare, and
  never bind devices yourself with `ASCEND_RT_VISIBLE_DEVICES`.
  - Correct: `npu-run python -m pytest tests/golden -v`
  - Wrong: `python -m pytest tests/golden -v`
  - Wrong: `ASCEND_RT_VISIBLE_DEVICES=3 python ...`
- Always use the default device: pass `-d 0` for real-device runs. The pool
  maps the assigned physical card to logical device 0, so code never needs to
  know (or choose) a physical device id.
- Simulator runs (`-p a2a3sim`, `-p a5sim`) do not touch the NPU and do not
  need `npu-run`.
- When all cards are busy, `npu-run` waits in place (polls every 1 s) and then
  proceeds as soon as a card is released. Waiting is normal — do not work
  around it. `npu-run --status` shows who holds each card; `npu-run --report`
  shows pool utilization.
- Use `npu-run --timeout <seconds> <command>` to bound the wait for
  long-running batches (exits with code 124 on timeout), and
  `npu-run --need N <command>` for multi-card tests.

## Preferred Commands

```bash
# Run an example on the simulator (no NPU involved, no npu-run needed)
python examples/beginner/hello_world.py -p a2a3sim

# Run a model on a real NPU (card assigned by the npu-run pool)
npu-run python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0

# Run golden harness unit tests on a real NPU
npu-run python -m pytest tests/golden -v

# Run repository lint checks (CPU only)
python tests/lint/check_headers.py
ruff check .
```

Every executable kernel or model script generally accepts
`-p {a2a3,a2a3sim,a5,a5sim}` and `-d <device_id>`. For real-device runs under
`npu-run`, always pass `-d 0`: the process sees exactly one card (the one the
pool assigned), exposed as logical device 0.

## Repository Map

- `examples/`: self-contained kernels for learning and reference patterns
- `models/`: end-to-end LLM kernels, one flat directory per model build
  (`<family>_<version>[_<quant>]`, e.g. `deepseek_v4_flash_mtp`)
- `golden/`: compile/run/validate harness against torch references
- `tests/`: lint checks and golden harness unit tests
- `docs/`: coding style, compile/runtime workflow, performance tuning,
  validation, examples, models, precision tuning, and debugging references
- `.claude/skills/`: task-specific execution workflows that reference `docs/`
