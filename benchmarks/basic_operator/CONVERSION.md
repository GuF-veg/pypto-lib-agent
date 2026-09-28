# Converting one model operator into a benchmark exercise

One operator = one exercise file = one agent. Follow this checklist exactly; the
gates at the bottom are mechanical and must all pass before you report success.

Scope is **single-NPU exercises only**: operators whose golden wiring needs more than
one card (`pld` / `DistributedConfig` device entries) are out of this batch -- an
exercise that cannot run on one device cannot be scored per candidate on a shared host.

Reference pair -- read both before touching anything:

- `benchmarks/basic_operator/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py` (accepted exercise, verified on device)
- `models/deepseek_v4_flash_mtp/decode_sparse_attn_swa.py` (the same operator, unstripped)

## Deliverable

`benchmarks/basic_operator/<model>/<file>.py`, where `<file>` is the model source
file name (`kda_conv.py` and `mla_prolog.py`/`mla_epilog.py` split into
`kda_conv_{prefill,decode}.py`, `mla_epilog_{prefill,decode}.py`,
`mla_prolog.py` + `mla_prolog_absorb_query.py`).

## CSV columns

`docs/models/golden-verified-basic-operators.csv` gives
`model,file,operator,kind,line,badge,golden_function,wiring_entry,wiring_entry_file,wiring_kind`.

- `proxy-test` (`◉`) -- the operator is the only kernel called by `wiring_entry`,
  and that entry owns the `run(fn=…, golden_fn=…)` wiring. Stub the operator, keep
  the entry **verbatim**.
- `twin-test` (`⊕`) -- the source binds one undecorated body twice:
  `X = pl.jit.inline(_impl)` and `X_test = pl.jit(_impl)`. Rebuild it as two
  functions: `@pl.jit.inline def X(<_impl's exact signature>)` holding the stub, and
  `@pl.jit def X_test(<same signature>)` calling `X` and returning the output.
  Delete `_impl` and both alias lines. If `_impl` opens with `pl.bind_dynamic(…)` or
  reads `pl.tensor.dim(…)` for the ABI, move those lines into `X_test` -- nothing else.
- `self` (`⊕`) -- the operator *is* the entry (`@pl.jit`): stub the entry itself, do
  not invent a wrapper.

## Rules

1. Copy the source file, then edit. Lines 1-8 (the license header) stay byte-identical.
2. Module docstring: one title line, then `Copied from models/<model>/<file>.py`
   (the checker parses this line), then an `EXAM:` paragraph naming the kernel under
   test, the golden function and the harness entry. Delete every design/strategy
   comment -- they are answer hints.
3. Bootstrap exactly like the reference: `REPO_ROOT = next(parent for parent in
   Path(__file__).resolve().parents if (parent / "golden" / "__init__.py").is_file())`
   then `sys.path.insert(0, str(REPO_ROOT))`, then module-level `import torch`,
   `import pypto.language as pl`, `from golden import …` (only the names used).
   No import may resolve under `models/`; drop bare-sibling imports,
   `models.<pkg>.*` dotted imports and any `sys.path.insert(…, parents[N])`.
4. A module-level constant survives **only** if one of these still reads it:
   (a) the stub's parameter annotations, (b) the kept entry body, (c) the golden
   function and any vendored helper, (d) the tensor-spec builders and `__main__`.
   Everything else -- tiling knobs, task counts, padding rounds -- is deleted,
   together with any `assert` whose operands are all deleted. Config values become
   literals under `# model config (<PRESET> preset of config.py)`; keep derived
   expressions in their original form (`NOPE_DIM = HEAD_DIM - ROPE_DIM`).
   `pl.dynamic("NAME")` bindings stay with byte-identical strings (compiled ABI).
5. Stub: keep the decorator and the FULL signature; the body is an EXAM docstring
   plus `return <output parameter>` -- nothing else. Do not add a placeholder task or
   a zero-fill: statements in an `@pl.jit.inline` body lower into the *orchestration*
   scope, where `pl.system.task_dummy` and bare `pl.full` stores are rejected by
   codegen. No renames, reordering or annotation edits.
6. The **executable** torch golden and tensor-spec code stays LINE-FOR-LINE
   identical: the checker diffs those lines against the model file, and hoisting
   `import torch` (or a sibling import) is the only permitted edit. No reformatting,
   renaming or simplifying. **Prose is the exception** -- docstrings and comments are
   excluded from that diff and must instead be scrubbed: the checker forbids any
   deleted-constant name anywhere in the file, comments included, so a source
   sentence like "the `SPARSE_BLOCKS` padding is exercised, rows span one to four
   sparse blocks" has to be rewritten while the fixture behaviour stays the same.
7. torch helpers from sibling modules are copied **by name** under
   `# fixture helpers (copied from models/<model>/<module>.py)` -- the checker
   byte-compares against that path. Never copy a module that contains `@pl.jit`
   kernels (`utils.py` is fine, `sample.py`/`prefill_metadata.py`/`kda_projection.py`
   are not: extract the torch functions only) and confirm the extracted text has no
   `pl.` call. Duplicated code across two exercises is expected; cross-file imports
   are not.
8. Exactly one kernel under test: delete any other operator in the file with its
   golden, entry and wiring.
9. `__main__`: keep the source's flags, mode loops and `run(...)` wiring (same
   `rtol`/`atol`/`compare_fn`), add `--golden-data`, `--save-data`, `--out-dir`
   exactly as in the reference, resolve user paths to absolute **before**
   `os.chdir(REPO_ROOT)`, pass `save_data=args.save_data`, and pass
   `save_kernels=True, save_kernels_dir=out_dir` inside `config=dict(…)` when
   `--out-dir` is given. Drop debug prints that dump tiling constants.
10. Delete `# ci:` markers, `__all__`, pytest helpers, benchmark writers, and any
    indented `__main__` guard (`if __name__ == _SCRIPT_ENTRY_POINT:` becomes the plain form).

## Model-tree quirks already hit by this batch

- When the golden or the spec builder reads a config OBJECT (`M.num_experts_per_tok`,
  `M.compress_rope_theta`, ...), rule 6 forbids editing that code, so define a local
  read-only stand-in next to the inlined literals and comment it clearly::

    # model config stand-in: only the fields the golden and the specs read
    class M:
        num_experts_per_tok = 6

  Use a `dataclasses`/`namedtuple` form when several fields are needed. Never import
  the model's `config` module to get `M`.
- A kernel declared `-> pl.Scalar[pl.TASK_ID]` returns that task-id parameter, not a
  tensor; the stub still ends with a single `return`, matching the declared type.
- Preserve a source `@pl.jit.inline(auto_scope=False)` decorator verbatim.

- Out-dirs must be model-qualified (`build_output/<model>__<file>.restore`): the same
  file name exists in several model trees, and a shared directory lets a concurrent
  agent overwrite your golden fixture mid-run.

- If `__main__` loops over modes or cases (decode/prefill, several `DynamicCase`s),
  narrow the DEFAULT to one deterministic selection: `--save-data` writes into a
  single `<out-dir>/data`, so a second case would overwrite the replay fixture of
  the first. `deepseek_v4_flash_mtp/rmsnorm.py` is the worked example.
- When the origin's spec builder opens with a multi-line `from utils import (
    a,
    b,
)` block, the sanctioned edit is to **delete that statement** (the helpers are
  already module-level functions in your file, so the names resolve). Never re-point
  it at another root: `from golden import <helper>` plus a `setattr(golden, ...)`
  monkey-patch is now a hard checker failure -- the harness package is shared state.
- If the origin opens with `from __future__ import annotations`, keep that line in
  the exercise (it is legal only as the first statement, and `restore_exam.py emit`
  deliberately does not re-inject it).
- `glm5_3_flash` model dirs contain a local `golden.py` that shadows the repo-root harness; `restore_exam.py emit` appends the model dir
  to `sys.path` for that reason. Do not add extra `sys.path` lines to the exercise.

> Scope: single-NPU exercises only. Operators whose golden wiring needs more than one
> card (`pld` / `DistributedConfig` entries) are deliberately excluded from this batch.

## What `tools/check_exam_stripping.py` enforces

1. import roots (stdlib / torch / numpy / pypto / golden only);
2. every `pl`/`pld` **call** lives in a `pl.jit` function or is a module-level
   `pl.dynamic` binding -- nothing leaks into the torch side;
3. no dead constant, and no constant the origin used only inside the kernel body,
   anywhere in the file, comments included (signatures and `pl.dynamic` bindings are
   interface, so they are exempt);
4. no line that belongs *only* to the deleted kernel body may survive -- verified by
   planting a real kernel line into a clean exercise, which the checker rejects;
5. every shared function's executable lines match the origin (imports, docstrings and
   comments excluded);
6. harness scaffolding: license header, REPO_ROOT bootstrap, `os.chdir`, and the
   `--golden-data` / `--save-data` / `--out-dir` flags.

## Gates

```bash
# from the repository root
source $(conda info --base)/etc/profile.d/conda.sh && conda activate pypto
export PYPTO_GOLDEN_NUM_THREADS=8
E=benchmarks/basic_operator/<model>/<file>.py
conda run -n pypto python tools/check_exam_stripping.py $E          # -> ok
conda run -n pypto python tools/restore_exam.py check $E            # -> shared constants verified, rc 0
conda run -n pypto ruff check --config ruff.toml $E                 # -> clean
conda run -n pypto python tools/restore_exam.py emit $E             # -> prints the two runs below
```

Then the two device runs on your assigned card `$CARD` (the emitter prints them
verbatim; keep the `flock` -- it serializes exercises that share a card):

```bash
flock -w 7200 /tmp/npu$CARD.lock python build_output/exam_verify/<model>__<file>.restore.py \
    -p a2a3 -d $CARD --out-dir build_output/<model>__<file>.restore --save-data   # MUST print [RUN] PASS, rc 0
flock -w 7200 /tmp/npu$CARD.lock python $E -p a2a3 -d $CARD \
    --out-dir build_output/<model>__<file>.stub --golden-data build_output/<model>__<file>.restore/data
                                                                          # MUST compile, then FAIL, rc 1
rm -f build_output/exam_verify/<model>__<file>.restore.py
rm -rf build_output/<model>__<file>.stub
```

The restore run is the only proof that the exercise is winnable and that your
inlined literals, vendored helpers and kept entry are faithful. The stub run is the
only proof that the implementation is gone, and it has **two acceptable shapes**:

- `[RUN] FAIL` with a per-element mismatch table (`actual=0`) when the retained
  entry submits tasks of its own -- as the reference exercise's `swa_valid_bias`
  scope does;
- a clean `[RUN] compile done` followed by a runtime fault (`chip run lane is
  poisoned`, `507018`, or an equivalent AICPU error) when the entry is only a
  `bind_dynamic` prologue plus the call to the now-empty kernel, because an image
  with zero tasks cannot execute.

Both give rc 1 and neither can be mistaken for a pass. Only a stub run that reaches
`[RUN] PASS` disqualifies the exercise. Say in your report which of the two you got.


## Failure ladder -- never report a pass you did not get

| symptom | meaning | action |
| --- | --- | --- |
| restore run FAIL or traceback | conversion defect (literal value, missing helper, damaged golden, entry lost a `bind_dynamic`) | fix and re-run |
| restore run FAIL and the model source also FAILs on the same card | the source case is broken on this host | report `blocked-source-fail` with both commands and outputs; do not ship a broken exercise |
| stub run dies during `[RUN] compile` | interface, `pl.dynamic` name or entry damaged | fix |
| stub run `[RUN] PASS` | implementation still present | strip further |
| allocation error / hang | card contention | retry once alone; if it persists report `skipped-busy` |

Never switch to another card, never use `-p a2a3sim` (it passes cases that fail on
device and silently compile-only-skips some), never report a numerical pass that
the log does not show.

## Off limits

`models/`, `golden/`, `docs/`, `tests/`, `tools/`, `benchmarks/basic_operator/manifest.csv`,
any other agent's exercise file, and every git write operation.
