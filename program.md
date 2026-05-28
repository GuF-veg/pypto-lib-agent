# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Create the branch**: `git checkout -b autoresearch` from current branch.
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
- Modify the script `models/qwen3/32b/qwen3_32b_decode.py` — this is the only file you can edit.

**What you CANNOT do:**
- Modify the evaluation harness.

**The goal is simple: get the lowest runtime.** You need to oprimize the performance of Qwen3Decode operator. The only constraint is that the code should be successfully compiled and runs without crashing and finishes with result checking succeed.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1 ms runtime improvement that adds 20 lines of hacky code? Probably not worth it. A 0.1 ms runtime improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the script as is.

## Output format

Once you run the script with `pprofile models/qwen3/32b/qwen3_32b_decode.py`, it prints a summary like this:

```
[npu-lock] 获取设备 6 的锁 (无超时)...
[npu-lock] 已获取设备 6 的锁 (pid=848507)
[RUN] compile ...
2026-05-28 14:06:27.775 I | [perf_hint] 50 hints across 21 sites; see build_output/Qwen3Decode_20260528_140626/report/perf_hints.log
[RUN] compile done (1.52s)
[RUN] generate inputs ...
[RUN] generate inputs done (6.39s)
[RUN] compute golden ...
[RUN] compute golden done (8.51s)
[RUN] runtime ...

✓ Conversion complete
  Input:  /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260528_140626/dfx_outputs/l2_perf_records.json
  Output: /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260528_140626/dfx_outputs/merged_swimlane_20260528_140647.json

To visualize: Open https://ui.perfetto.dev/ and drag in /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260528_140626/dfx_outputs/merged_swimlane_20260528_140647.json

==============================================================================================================
Task Statistics by Function
  Exec = kernel time on AICore; Latency = dispatch->finish (incl. head OH + Exec + tail OH)
==============================================================================================================
Func_ID  Func_Name    Count   Avg Exec(us)  Avg Latency(us)   Exec%   Avg Head OH(us)  Avg Tail OH(us)
--------------------------------------------------------------------------------------------------------------
0        rmsnorm          1          24.70            31.10   79.4%              1.50             4.90
1        q_proj          32          93.54           121.56   76.9%             25.45             2.58
2        kv_proj          4         105.40           199.94   52.7%             92.96             1.57
3        rope_kv_cache    16           9.90            21.00   47.2%              0.55            10.55
4        qk_matmul       64          28.04            43.73   64.1%             12.06             3.63
5        softmax         64          15.82            21.34   74.1%              0.55             4.98
6        sv_matmul       64          28.49            56.37   50.5%              9.84            18.04
7        online_softmax    64           7.63            19.42   39.3%              0.49            11.29
8        out_proj_residual_aic    16         171.06           177.97   96.1%              0.62             6.30
9        out_proj_residual_aiv    32         170.68           178.51   95.6%              0.49             7.35
10       post_rmsnorm     1          32.24            37.86   85.2%              0.42             5.20
11       gate_proj      100          89.94           176.77   50.9%             84.87             1.96
12       up_proj        100          94.42           172.03   54.9%             75.67             1.94
13       silu           100           2.08             5.77   36.1%              0.47             3.22
14       down_proj_residual__windowed_aic    16         416.14           424.95   97.9%              0.50             8.30
15       down_proj_residual__windowed_aiv    32         415.81           425.60   97.7%              0.46             9.33
--------------------------------------------------------------------------------------------------------------
TOTAL                   706       55555.74         78544.62

Total Test Time: 2074.76 us (from earliest dispatch to latest finish)

--- Task execution vs Scheduler overhead ---
  Per-task (all):  Avg Exec = 78.69 us,  Avg Latency (dispatch->finish) = 111.25 us,  Exec/Latency = 70.73%
  (Latency = dispatch→finish; Exec = AICore kernel time per task)
==============================================================================================================

=== Scheduler Overhead Deep Dive ===
Perf data:  /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260528_140626/dfx_outputs/l2_perf_records.json

==========================================================================================
Part 1: Per-task time breakdown (from perf profiling data)
==========================================================================================
Total tasks: 706
Wall-clock:  2074.8 us

  Component                             Total (us)  Avg/task (us)  % of Latency
  ------------------------------------------------------------------------------
  Kernel Exec (end-start)                  55555.7          78.69         70.7%
  Head OH (start-dispatch)                 18814.1          26.65         24.0%
  Tail OH (finish-end)                      4174.7           5.91          5.3%

==========================================================================================
Part 2: AICPU scheduler loop breakdown
  3 scheduler threads
==========================================================================================

  Thread       Loops  Completed   Tasks/loop  Total (us)
  ------------------------------------------------------
  T0            5160         79          0.0      1901.0
  T1            2438        109          0.0      2031.3
  T2            4968         78          0.0      2021.1
  SUM          12566        266          0.0      5953.4

  Phase                                               Total (us) % of total  Avg/task (us)
  -----------------------------------------------------------------------------------------
  Complete (poll handshake, resolve deps)                  736.5      12.4%           2.77
  Scan (update perf header)                                  0.0       0.0%           0.00
  Dispatch (pop queue, build payload, flush)               683.0      11.5%           2.57
  Idle (spinning, no progress)                            4533.9      76.2%          17.04

  Fanout / Fanin: (deps.json not provided — pass --deps-json or rerun with --enable-dep-gen)

  Pop: hit=410, miss=35973, hit_rate=1.1%

==========================================================================================
Part 3: Tail OH distribution & cause analysis
==========================================================================================

  Tail OH distribution (N=706):
    P10        0.8 us
    P25        1.4 us
    P50        2.8 us
    P75        6.2 us
    P90       11.6 us
    P95       14.4 us
    P99       83.1 us
    Max:      92.7 us
    Mean:      5.9 us

  Avg scheduler loop iteration: 0.5 us (approx avg polling interval per loop)

  Avg Tail OH = 5.9 us ~= 12.5 x avg loop iteration (0.5 us)
  -> On average, a completed task waits ~12.5 loop iterations before being detected

  Key insight: Complete phase consumes ~12% of scheduler CPU.
  DAG stats unavailable (no deps.json); cannot attribute complete-phase cost further.
==========================================================================================
Swimlane JSON written to: /data/gufeng/project/pypto-lib/build_output/Qwen3Decode_20260528_140626/dfx_outputs/merged_swimlane_20260528_140647.json
[RUN] runtime done (5.79s)
[RUN] validate ...
[RUN]   'out' PASS  shape=(16, 8192) dtype=torch.bfloat16
[RUN] validate done (0.03s)
[RUN] PASS (22.23s)
[npu-lock] 已释放设备 6 的锁
=== 任务完成 (exit=0) ===
```

The first thing you need to care is at last it should report a `[RUN] PASS`. 
This means the result is correct.
The most important metric is the runtime of mk_gat operator. You can extract the key metric by:

```bash
python megakernels/scripts/gat.py -b | grep -oP 'mk_gat\s+:\s+\K[\d.]+'
```

## Logging results

When an experiment is done, log it to `results/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

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
commit	val_bpb	status	description
a1b2c3d	23.997	keep	baseline
b2c3d4e	20.993	keep	optimize...
c3d4e5f	18.005	discard	add...
d4e5f6g	0.000	crash	deadlock
```

## Oprimization tips

You can refer to `src/gcn/` directory, which is a highly oprimized GNN kernel.
You can use `ncu` (NVIDIA Nsight compute) instruction to profile the kernel and find the bottlenecks.
When necessary, conduct an online search for optimization ideas — after failing to make effective progress in three consecutive discussions, you need to search online for new optimization ideas.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch`) created from the human development branch (`5090`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune the codes in `src/gat` with an experimental idea by directly hacking the code, before you coding, please deep think and do the detailed plan.
3. Run the command line instruction: `conda activate dev && source setup.py && make -C src/gat` to compile the code, you should make sure the compilation is successful.
4. Run the validation: `python megakernels/scripts/gat.py` to make sure the results is correct, if it is not correct, you need to go back to step 3, debug and recompile the code.
5. Run the benchmark: `python megakernels/scripts/gat.py -b | grep -oP 'mk_gat\s+:\s+\K[\d.]+'` to get the runtime of your code.
6. The running process should be very fast, so if the script doesn't finish for a long time, it might be a deadlock in your code, and you need to kill the process and debug.
7. Record the results in the tsv.
8. If the runtime improved (lower), write your optimization method in `docs/GAT/optimization_log.md` and commit your changes in git.
9. If the runtime is worse, discard all your changes and reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take at most 5 minutes total. If a run exceeds 5 minutes, there must be a deadlock bug, kill it and treat it as a failure.

**Crashes**: If a run crashes (deadlock or compile error), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just discard it.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely*. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the performance of mk_gat is significantly better than pytorch_compiled version.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
