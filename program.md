# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Create the branch**: `git checkout -b autoresearch` from current branch (if already on this branch, ignore this).
2. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `docs/` — The documents of pypto-lib.
   - `models/qwen3/32b/qwen3_32b_decode.py` — the file you modify.
3. **Initialize results.tsv**: Create `results/results.tsv` with just the header row. The baseline will be recorded after the first run.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. 
You can run an experiment simply as: 
```bash
conda activate pypto
pprofile models/qwen3/32b/qwen3_32b_decode.py
```

**What you CAN do:**
- Modify the script `models/qwen3/32b/qwen3_32b_decode.py` (except the `golden_qwen3_decode` function) — this is the only file you can edit.

**What you CANNOT do:**
- Modify the evaluation harness.
- Modify the `golden_qwen3_decode` function.

**The goal is simple: get the lowest runtime.** You need to oprimize the performance of Qwen3Decode operator. The only constraint is that the code should be successfully compiled and runs without crashing and finishes with result checking succeed.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1 ms runtime improvement that adds 20 lines of hacky code? Probably not worth it. A 0.1 ms runtime improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the script as is.

## Output format

Once you run the script with `pprofile models/qwen3/32b/qwen3_32b_decode.py`, it prints a summary like this:

```
[RUN] compile ...
2026-05-29 09:48:05.350 I | [perf_hint] 50 hints across 21 sites; see build_output/Qwen3Decode_20260529_094804/report/perf_hints.log
[RUN] compile done (1.55s)
[RUN] generate inputs ...
[RUN] generate inputs done (5.67s)
[RUN] compute golden ...
[RUN] compute golden done (6.49s)
[RUN] runtime ...

✓ Conversion complete
  Input:  /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260529_094804/dfx_outputs/l2_perf_records.json
  Output: /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260529_094804/dfx_outputs/merged_swimlane_20260529_094826.json

To visualize: Open https://ui.perfetto.dev/ and drag in /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260529_094804/dfx_outputs/merged_swimlane_20260529_094826.json

==============================================================================================================
Task Statistics by Function
  Exec = kernel time on AICore; Latency = dispatch->finish (incl. head OH + Exec + tail OH)
==============================================================================================================
Func_ID  Func_Name    Count   Avg Exec(us)  Avg Latency(us)   Exec%   Avg Head OH(us)  Avg Tail OH(us)
--------------------------------------------------------------------------------------------------------------
0        rmsnorm          1          23.74            26.70   88.9%              0.98             1.98
1        q_proj          32          97.18           130.01   74.7%             26.57             6.26
2        kv_proj          4         104.88           202.95   51.7%             97.08             1.00
3        rope_kv_cache    16           9.78            19.94   49.0%              0.54             9.63
4        qk_matmul       64          22.71            44.89   50.6%             12.26             9.92
5        softmax         64          12.96            29.39   44.1%              0.46            15.96
6        sv_matmul       64          22.26            33.42   66.6%              3.22             7.94
7        online_softmax    64           6.16            11.01   56.0%              0.43             4.42
8        out_proj_residual_aic    16         164.52           171.17   96.1%              0.58             6.08
9        out_proj_residual_aiv    32         164.18           171.93   95.5%              0.47             7.27
10       post_rmsnorm     1          31.86            37.76   84.4%              0.66             5.24
11       gate_proj      100          90.21           169.95   53.1%             77.49             2.25
12       up_proj        100          88.96           168.77   52.7%             78.00             1.81
13       silu           100           2.13             5.62   38.0%              0.45             3.03
14       down_proj_residual__windowed_aic    16         412.85           418.37   98.7%              0.51             5.02
15       down_proj_residual__windowed_aiv    32         412.56           418.83   98.5%              0.46             5.81
--------------------------------------------------------------------------------------------------------------
TOTAL                   706       53666.98         75722.48


--- Task execution vs Scheduler overhead ---
  Per-task (all):  Avg Exec = 76.02 us,  Avg Latency (dispatch->finish) = 107.26 us,  Exec/Latency = 70.87%
  (Latency = dispatch→finish; Exec = AICore kernel time per task)
==============================================================================================================

=== Scheduler Overhead Deep Dive ===
Perf data:  /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260529_094804/dfx_outputs/l2_perf_records.json

==========================================================================================
Part 1: Per-task time breakdown (from perf profiling data)
==========================================================================================
Total tasks: 706
Wall-clock:  2032.3 us

  Component                             Total (us)  Avg/task (us)  % of Latency
  ------------------------------------------------------------------------------
  Kernel Exec (end-start)                  53667.0          76.02         70.9%
  Head OH (start-dispatch)                 17938.2          25.41         23.7%
  Tail OH (finish-end)                      4117.3           5.83          5.4%

==========================================================================================
Part 2: AICPU scheduler loop breakdown
  3 scheduler threads
==========================================================================================

  Thread       Loops  Completed   Tasks/loop  Total (us)
  ------------------------------------------------------
  T0            5073         75          0.0      1842.2
  T1            3554         93          0.0      1965.1
  T2            3138         98          0.0      1988.3
  SUM          11765        266          0.0      5795.6

  Phase                                               Total (us) % of total  Avg/task (us)
  -----------------------------------------------------------------------------------------
  Complete (poll handshake, resolve deps)                  711.4      12.3%           2.67
  Scan (update perf header)                                  0.0       0.0%           0.00
  Dispatch (pop queue, build payload, flush)               641.4      11.1%           2.41
  Idle (spinning, no progress)                            4442.7      76.7%          16.70

  Fanout / Fanin: (deps.json not provided — pass --deps-json or rerun with --enable-dep-gen)

  Pop: hit=404, miss=33754, hit_rate=1.2%

==========================================================================================
Part 3: Tail OH distribution & cause analysis
==========================================================================================

  Tail OH distribution (N=706):
    P10        0.9 us
    P25        1.5 us
    P50        2.9 us
    P75        5.8 us
    P90       11.7 us
    P95       15.9 us
    P99       73.9 us
    Max:      82.2 us
    Mean:      5.8 us

  Avg scheduler loop iteration: 0.5 us (approx avg polling interval per loop)

  Avg Tail OH = 5.8 us ~= 11.8 x avg loop iteration (0.5 us)
  -> On average, a completed task waits ~11.8 loop iterations before being detected

  Key insight: Complete phase consumes ~12% of scheduler CPU.
  DAG stats unavailable (no deps.json); cannot attribute complete-phase cost further.
==========================================================================================

Total Test Time: 2033.48 us (avg over 5 repeat runs after 2 warmup; min=2012.72 max=2059.32 std=15.54)
Swimlane JSON written to: /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260529_094804/dfx_outputs/merged_swimlane_20260529_094826.json
[RUN] runtime done (9.32s)
[RUN] validate ...
[RUN]   'out' PASS  shape=(16, 8192) dtype=torch.bfloat16
[RUN] validate done (0.03s)
[RUN] PASS (23.06s)
```

The first thing you need to care is at last it should report a `[RUN] PASS`. 
This means the result is correct.
The most important metric is the `Total Test Time`. 
Other indicators can assist you in performance optimization.

## Logging results

When an experiment is done, log it to `log/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	runtime	status	description
```

1. git commit hash (short, 7 chars)
2. runtime achieved (e.g. 1.234) — use 0.000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	runtime	status	description
a1b2c3d	23.997	keep	  baseline
b2c3d4e	20.993	keep	  optimize...
c3d4e5f	18.005	discard	add...
d4e5f6g	0.000	  crash	  deadlock
```

## Oprimization tips

You can refer to `models/` directory, which are highly oprimized model programs.
You can refer to `docs/` and `../pypto/docs/en` directories, which are documents related to pypto.
When necessary, conduct an online search for optimization ideas — after failing to make effective progress in three consecutive discussions, you need to search online for new optimization ideas.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch`) created from the human development branch (`main`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on, you maybe already on the autoresearch branch, then go to next step.
2. Tune the codes in `models/qwen3/32b/qwen3_32b_decode.py` with an experimental idea by directly hacking the code, before you coding, please use superpower to brainstorm and do the detailed plan.
3. Run the cmd instruction `rm -rf build_output/*` to clear the temp files.
4. Run the script: `pprofile models/qwen3/32b/qwen3_32b_decode.py` to get the result of your code. You need to check there should be `[RUN] PASS` in the output which indicates the correctness of your algorithm. You can use `grep -oP 'Total Test Time:\s+\K[\d.]+'` to the output to get the runtime, but other information in the output maybe helpful for you to do the optimization.
5. If the results don't pass, you need to debug by yourself. If the bug still cannot be resolved after multiple iterations, discard this modification and reset back to where you started.
6. If the runtime improved (Due to timing inaccuracies in program execution, if there is a performance improvement, you need to run repeated experiments to ensure that the improvement is reproducible.), write your optimization method in `docs/successful_optimization.md` and commit your changes in git.
7. If the runtime is worse, write your lessons learned in `docs/failed_optimization.md`, discard all your changes and reset back to where you started.
8. Record the results in the tsv.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run fails, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just discard it.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely*. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the performance of mk_gat is significantly better than pytorch_compiled version.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
