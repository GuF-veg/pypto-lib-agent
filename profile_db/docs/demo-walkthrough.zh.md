# pfdb 实机演示：DeepSeek-V4 CSA Decode 算子全流程

本演示用仓库中的真实大算子 `models/deepseek_v4_flash_dspark/decode_csa.py`（DeepSeek-V4 上下文并行 CSA decode 层，TP=1 单卡形态）走完 pfdb 的全部核心环节：**真机采集 → 入库 → 宏观体检 → 关键路径定位 → 单任务微归因 → 管线画像 → 可视化 → 调优闭环 → MCP agent 会话**。每一步都给出可直接复制的命令与真机原样输出（仅对含本机绝对路径的字段做截断脱敏），并说明这一步展示了系统的什么能力。

> 配套文档：各命令/查询的完整参数与语义见[使用手册](./usage-manual.zh.md)。
>
> **复现须知**：所有数值随运行漂移（设备状态、编译缓存、批次参数都会影响），task_id 也可能随图版本变化。复现时请以你自己的实跑输出为准，"跟着结构走"而不是"对数字"。

---

## 0. 演示亮点一览

| 环节 | 展示的能力 | 本次实测 |
|---|---|---|
| 入库 | 大图一次 ingest：72 任务 / 4271 物理行 / 242 依赖边 / 219 条编译器提示，14 秒完成、幂等可重放 | §4 |
| 宏观体检 | 两条引擎 60 核的分带占用、稀疏带确定性归因 | §5 |
| 关键路径 | observed CPM 25 个任务、最大 stall 排序 | §6 |
| 微归因 | FIN→dispatch→receive→start 四段分解、**时间戳证明的早派发（7/24 块）**、调度相流 | §7 |
| 管线画像 | qk_pv 融合核的 PMU：scalar 40.3% / mte2 25.0% | §8 |
| 可视化 | 60 核 × 4 ms 全景泳道 PNG，一图看清计算波次与空窗 | §9 |
| 调优闭环 | compare + 分层 bootstrap 置信区间 + trial 裁决记录 | §10 |
| Agent 通道 | 38 个 MCP 工具的完整会话 | §11 |
| 观测开销 | 干净采集 vs 全模态采集的 makespan 对比（3.98 ms vs 72.8 ms）——为什么裁决必须看 bench | §10 |

---

## 1. 前置条件

| 项 | 要求 | 本演示实测 |
|---|---|---|
| 环境 | conda `pypto`（PyPTO 运行时 + pfdb） | `pfdb --version` → `pfdb 0.3.0` |
| 设备 | 一张空闲 910B（`a2a3`） | `npu-smi info` 确认 device 0 空闲 |
| 算子 | `models/deepseek_v4_flash_dspark/decode_csa.py` | 暴露 `--tp {1,2,4}`、`--enable-chip-swimlane [{0..4}]`、`--enable-pmu`、`--enable-dump-args`、`--enable-scope-stats` |
| 编译 | 首次运行含编译，约 3 分钟；后续有缓存更快 | run A 167 s / run B 168 s |

> **为什么 `--tp 1`**：脚本默认 `--tp 2` 需要 2 张卡（`-d 0,1`）。单卡演示必须显式 `--tp 1`。

---

## 2. 准备工作区

```bash
conda activate pypto
cd <仓库根目录>

# 演示专属数据库（.pfdb/ 已 gitignore；删掉即重置）
export PFDB_PATH=$PWD/.pfdb/demo/profile.duckdb
mkdir -p .pfdb/demo
pfdb init
```

输出：

```text
pfdb initialized at …/.pfdb/demo/profile.duckdb (schema_version=7)
```

> **每个 shell 都要 `export PFDB_PATH`**。忘了就会写到默认路径，之后查出来是"空库"。

---

## 3. 采集（pfdb 之外，但决定了 pfdb 有什么数据）

pfdb 自己不采集。本演示需要 5 次真机运行：

### 3.1 Run A：干净计时采集（level-4 泳道）

```bash
python models/deepseek_v4_flash_dspark/decode_csa.py -p a2a3 -d 0 --tp 1 \
    --enable-chip-swimlane 4 2>&1 | tee capture_A.log
```

要点输出：

```text
[swimlane] chip swimlane enabled -> running the kernel twice (dep_gen perturbs timing, …):
[swimlane] run 1/2: capturing the task dependency graph (deps.json) in a subprocess; …
[swimlane] run 2/2: measuring clean per-task timing (this run's numbers are the ones reported).
Swimlane JSON written to: build_output/_jit_decode_csa_tp1_test_<hashA>/dfx_outputs/merged_swimlane_<ts>.json
[RUN] PASS (167.20s)
```

运行时自动跑两遍：第一遍子进程抓依赖图，第二遍干净计时。记下 `<hashA>`（本次为 `_jit_decode_csa_tp1_test_vh3vah7d`）。

### 3.2 Run B：全模态采集（swimlane + PMU + 作用域统计 + 参数转储）

```bash
python models/deepseek_v4_flash_dspark/decode_csa.py -p a2a3 -d 0 --tp 1 \
    --enable-chip-swimlane 4 --enable-pmu 2 --enable-scope-stats --enable-dump-args 2 \
    2>&1 | tee capture_B.log
```

记下 `<hashB>`（本次为 `_jit_decode_csa_tp1_test_9kt_mfsw`）。检查 B 的产物：

```bash
ls build_output/_jit_decode_csa_tp1_test_<hashB>/dfx_outputs/
```

```text
args_dump  chip_swimlane_records.json  deps.json  merged_swimlane_<ts>.json
name_map__jit_decode_csa_tp1_test_<hashB>.json  pmu.csv  profile_capture_manifest.json  scope_stats
```

### 3.3 三份独立无观测基准（裁决数字）

```bash
for i in 1 2 3; do
  PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 \
    python models/deepseek_v4_flash_dspark/decode_csa.py -p a2a3 -d 0 --tp 1 \
    > bench_run$i.log 2>&1
  grep effective_us bench_run$i.log | tail -1
done
```

本次实测（三次独立调用，每次 100 轮）：

```text
[RUN]   effective_us (100 rounds) min=3534.8 median=3648.2 mean=3668.7 max=3885.6
[RUN]   effective_us (100 rounds) min=3545.8 median=3663.1 mean=3692.2 max=3944.2
[RUN]   effective_us (100 rounds) min=3545.6 median=3677.6 mean=3697.2 max=3888.0
```

`PYPTO_BENCH_RAW=1` 让日志带上逐轮原始样本（`headline raw n=100 eff_us=[…]`），pfdb 的分层 bootstrap 置信区间需要它。

---

## 4. 入库

```bash
# Run A：干净采集 + 3 份独立 bench 分层
pfdb ingest build_output/_jit_decode_csa_tp1_test_<hashA>/dfx_outputs \
    --program decode_csa_tp1 --platform a2a3 --device 0 --no-prune \
    --bench-log bench_run1.log --bench-log bench_run2.log --bench-log bench_run3.log \
    --notes "decode_csa run A: level-4 swimlane + 3 independent bench strata" --tags demo

# Run B：全模态采集
pfdb ingest build_output/_jit_decode_csa_tp1_test_<hashB>/dfx_outputs \
    --program decode_csa_tp1 --platform a2a3 --device 0 --no-prune \
    --notes "decode_csa run B: swimlane + pmu + scope_stats + dump_args" --tags demo

pfdb list
```

真实输出：

```text
ingested run 1 (program=decode_csa_tp1 level=4 mode=link)
  tasks=72 task_rows=4271 edges=242 artifacts=5 perf_hints=219 memory=0 pmu=0 makespan=3984.040us
  ingest_wall_ms=13386.781 ingest_cpu_ms=24472.001 peak_rss_bytes=1092128768 …
ingested run 2 (program=decode_csa_tp1 level=4 mode=link)
  tasks=72 task_rows=4271 edges=242 artifacts=8 perf_hints=219 memory=0 pmu=38439 makespan=72813.520us
  ingest_wall_ms=112442.024 …
RUN bench_mean_us=3685.488 device_id=0 edges=242 level=4 makespan_us=3984.04 platform="a2a3" program="decode_csa_tp1" rank="single" retained=true rows=4271 run_id=1 tasks=72 evidence=measured
RUN bench_mean_us=3686.013 device_id=0 edges=242 level=4 makespan_us=72813.52 platform="a2a3" program="decode_csa_tp1" rank="single" retained=true rows=4271 run_id=2 tasks=72 evidence=measured
```

这一步展示了什么：

- **大图秒级入库**：4271 行物理时序 + 242 条边 + 衍生层（1594 个密度带、2900 个空闲段、双关键路径）在一个事务里落表；
- **`--program` 归一化**：两次采集的 `_jit_<hash>` 目录名不同，统一传 `--program decode_csa_tp1` 才能让后面的 compare 认它们是同一程序；
- **注意 run 2 的 makespan=72813µs**：是 run 1（3984µs）的 18 倍——不是代码变慢，是 pmu/dump_args/scope_stats 的观测开销污染了计时。**时序结论看 run 1，模态内容查 run 2**；两 run 的 bench 都约 3686µs，证明真实性能没变。这就是"bench 裁决、泳道解释"的最佳活教材；
- 幂等：任何一次 ingest 原样重跑，行数不变。

---

## 5. 宏观体检（Z0/Z1）

### 5.1 顶线指标

```bash
pfdb query overview --run-id 1
```

```text
RUN clock_freq_hz=50000000 device_id=0 git_commit="326322e…" git_dirty=true level=4 num_cores=60 platform="a2a3" program="decode_csa_tp1" rank="single" retained=true run_id=1 evidence=measured
METRIC artifacts=5 bench_max_us=3945.5 bench_mean_us=3685.488 bench_median_us=3671.0 bench_min_us=3515.0 bench_rounds=300 cpm_us=3753.3 edges=242 idle_gaps=2900 makespan_us=3984.04 raw_span_us=3980.84 run_id=1 task_rows=4271 tasks=72 time_bands=1594 evidence=measured
RESOURCE cores=20 engine="aic" run_id=1 evidence=measured
RESOURCE cores=40 engine="aiv" run_id=1 evidence=measured
```

解读：60 核（20 AIC + 40 AIV）；CPM 3753µs 占 makespan 3984µs 的 **94%**——这个算子的调度已经相当紧凑，优化空间主要在关键路径本身而非调度空洞。 `level=4` 确认全部 Z4 查询可用。

### 5.2 密度带

```bash
pfdb query density --run-id 1 --bands 20
```

```text
BAND band_idx=0 busy_cores=20 drain_tail=false engine="aic" run_id=1 sparse=false t0_us=75.72 t1_us=275.72 task_ids=["8589934597"] total_cores=20 evidence=measured
BAND band_idx=1 busy_cores=0 drain_tail=false engine="aic" run_id=1 sparse=true t0_us=275.72 t1_us=475.72 task_ids=[] total_cores=20 evidence=measured
BAND band_idx=2 busy_cores=20 … t0_us=475.72 t1_us=675.72 task_ids=["8589934614","8589934618",…] …
…
BAND band_idx=18 busy_cores=20 drain_tail=true engine="aic" run_id=1 sparse=false t0_us=3675.72 t1_us=3875.72 task_ids=["12884901902",…] total_cores=20 evidence=measured
BAND band_idx=19 busy_cores=0 drain_tail=true engine="aic" run_id=1 sparse=false t0_us=3875.72 t1_us=4060.72 task_ids=[] total_cores=20 evidence=measured
TRUNCATED first_dropped_index=20 remaining=20 limit=4096 hint="retry --budget 32768"
```

解读：AIC 引擎大部分满载（20/20），275–475µs 有一条明显空带，末尾是 drain 尾。默认预算下显示分桶被 TRUNCATED，`hint` 教你加预算。

### 5.3 稀疏带归因

```bash
pfdb query sparse_regions --run-id 1 --top-k 3
```

```text
SPARSE busy_cores=0 engine="aic" fin_us=508.4 kind="ready_starved" lagging_producer="8589934603" run_id=1 stored_band_idx=26 t0_us=205.72 t1_us=210.72 total_cores=20 evidence=proven
SPARSE busy_cores=0 … lagging_producer="8589934603" … stored_band_idx=27 t0_us=210.72 t1_us=215.72 … evidence=proven
SPARSE busy_cores=0 … lagging_producer="8589934603" … stored_band_idx=28 t0_us=215.72 t1_us=220.72 … evidence=proven
```

解读：最空的存储带全部 `ready_starved`（上游没喂上）且**点名同一个滞后生产者** `8589934603`——这是一个 dummy（占位）任务。归因带 `evidence=proven`：由记录的确定性结构推出，不是猜测。

---

## 6. 关键路径（Z4）

```bash
pfdb query critical_path --run-id 1 --kind observed
```

PATH 行自带任务名/family/引擎，无需二次查询。节选（按 stall 排序后的前 8 名，可用 `--format json` 自行加工）：

```text
PATH … stall_us=14.5  task_id="8589934625" name="idx_qr_proj_matmul"  gap=1.22 kind="data-wait" wall=213.32 busy=206.96 …
PATH … stall_us=9.1   task_id="8589934646" name="qk_pv_aic+qk_pv_aiv" gap=7.58 kind="data-wait" wall=799.7  busy=798.38 …
PATH … stall_us=7.64  task_id="8589934600" name="split_pre_post"      gap=1.7  kind="data-wait" wall=239.72 busy=232.8  …
PATH … stall_us=7.52  task_id="8589934637" name="indexer_score_leaf_wave…" gap=6.9 kind="data-wait" wall=143.94 busy=141.94 …
PATH … stall_us=7.24  task_id="8589934629" name="qr_rope"             gap=5.6  kind="data-wait" wall=107.68 busy=104.5  …
PATH … stall_us=7.14  task_id="12884901915" name="proj_a_mm"          gap=6.54 kind="data-wait" wall=137.36 busy=132.88 …
PATH … stall_us=6.74  task_id="8589934632" name="qr_hadamard_matmul"  gap=4.38 kind="data-wait" wall=62.46 busy=59.82  …
PATH … stall_us=4.48  task_id="8589934614" name="kv_score_proj"       gap=3.76 kind="data-wait" wall=202.52 busy=200.48 …
```

这一步展示了什么：25 个任务构成观测关键路径；`qk_pv`（flash attention 主融合核）wall 799.7µs 是**最大的时间贡献者**（占 makespan 20%）；最大 stall 只有 14.5µs——再次印证"调度空洞很小，重心在核本身"。`--kind static` 可以交叉验证静态最长路径。

---

## 7. 单任务微归因（Z4 下钻）

### 7.1 最大耗时任务：qk_pv

```bash
pfdb query task --run-id 1 --task-id 8589934646
```

```text
TASK block_num=24 busy_us=798.38 early_dispatch_flag=true engine="mixed" family="qk_pv" kernel_ids=[43,44,44] max_end_us=2633.68 max_finish_us=2634.32 min_dispatch_us=1834.62 min_receive_us=1834.94 min_start_us=1835.3 name="qk_pv_aic+qk_pv_aiv" num_rows=72 on_cpm_observed=true on_cpm_static=true run_id=1 scope="auto" task_id="8589934646" wall_us=799.7 evidence=measured
```

72 个物理行铺在 AIC+AIV 两类核上（`engine="mixed"`）。它的物理行细节与族内对比：

```bash
pfdb query rows     --run-id 1 --task-id 8589934646 | head -3
pfdb query why_long --run-id 1 --task-id 8589934646
```

```text
ROW core_index=0 dispatch_us=1834.9 end_us=2268.5 finish_us=2269.2 receive_us=1835.18 row_index=17 run_id=1 start_us=1835.66 task_id="8589934646" evidence=measured
ROW core_index=1 dispatch_us=1839.82 end_us=2229.84 …
ROW core_index=2 dispatch_us=1834.62 end_us=2245.82 …
LONG busy_us=798.38 family="qk_pv" family_median_us=798.38 family_rank=1 family_tasks=1 max_row_us=444.08 min_row_us=353.38 name="qk_pv_aic+qk_pv_aiv" num_rows=72 run_id=1 task_id="8589934646" wall_us=799.7 evidence=measured
```

行级 min/max（353–444µs）给出核间散差。

它的上游依赖（含三种边来源）：

```bash
pfdb query deps --run-id 1 --task-id 8589934646 --direction in
```

```text
DEP arg="-1" consumer_dtype="" consumer_shape=[] … flags=["wait","retain"] pred="8589934642" run_id=1 source="explicit" succ="8589934646" tensor_id="" evidence=measured
DEP arg="" consumer_dtype="INT64" consumer_shape=[256] … source="creator" …
DEP arg="1" consumer_dtype="BFLOAT16" consumer_shape=[32768,512] … source="creator" …
DEP arg="1" consumer_dtype="BFLOAT16" consumer_shape=[32768,512] … source="tensormap" …
```

### 7.2 最大 stall 任务：早派发被时间戳证明

```bash
pfdb query why_late       --run-id 1 --task-id 8589934625
pfdb query early_dispatch --run-id 1 --task-id 8589934625
```

```text
STALL dispatch_us=846.92 dispatch_wait_us=2.16 fin_detect_us=-2.12 gap_us=1.22 ready_us=849.04 receive_us=849.08 run_id=1 start_us=850.26 start_wait_us=1.18 task_id="8589934625" upstream_depth=6 evidence=measured
EARLY proven_blocks=7 ready_us=849.04 run_id=1 status="partial" task_id="8589934625" tol_us=0.04 total_blocks=24 evidence=proven
```

这一步展示了什么：

- `why_late` 的恒等式 `gap = fin_detect + dispatch_wait + start_wait`（1.22µs 的 gap 全部分解到位）；`upstream_depth=6` 说明它背后有 6 层生产者链；
- `early_dispatch` 给出**四态证明**：24 块中 7 块被时间戳证明提前派发（`status="partial"`，`evidence=proven`，容差 2 拍 = 0.04µs）。"代码里标了 allow_early_resolve" 和 "硬件上真的提前派发了" 是两回事，pfdb 用时间戳区分它们。

### 7.3 调度相流

```bash
pfdb query scheduler --run-id 1 --task-id 8589934625
```

```text
SCHED kind="dispatch" lane=0 loop_iter=436 run_id=1 t0_us=844.52 t1_us=846.64 tasks_processed=3 evidence=measured
SCHED kind="early_dispatch" lane=0 loop_iter=436 run_id=1 t0_us=846.7 t1_us=849.54 tasks_processed=7 evidence=measured
SCHED kind="complete" lane=0 loop_iter=437 …
SCHED kind="release" lane=0 loop_iter=441 …
SCHED kind="complete" lane=1 loop_iter=552 run_id=1 t0_us=839.4 t1_us=847.4 tasks_processed=21 evidence=measured
…
TRUNCATED first_dropped_index=35 remaining=50 limit=4096 hint="retry --budget 32768"
```

任务窗口（±1µs）重叠了整整 85 条调度相（两条 lane 的 dispatch/complete/release/early_dispatch 流）——大任务周边调度器在做什么，一目了然。`--budget 32768` 可看全。

---

## 8. 管线画像：PMU（查 run 2）

```bash
pfdb query pmu --run-id 2 --task-id 8589934646
```

```text
MODALITY entry_count=38439 modality="pmu" parser_state="parsed" rel_path="dfx_outputs/pmu.csv" request_value="2" requested=true run_id=2 size_bytes=285922 state="available" evidence=measured
PMU_SUMMARY counters=9 measurements=648 run_id=2 samples=72 task_id="8589934646" evidence=measured
PMU counter="scalar_busy_cycles" ratio=0.402731 run_id=2 samples=72 task_id="8589934646" total_cycles=46384836.0 value=18680628.0 evidence=measured
PMU counter="mte2_busy_cycles"  ratio=0.2504   … value=11614763.0 evidence=measured
PMU counter="vec_busy_cycles"   ratio=0.196556 … value=9117213.0  evidence=measured
PMU counter="mte3_busy_cycles"  ratio=0.108475 … value=5031576.0  evidence=measured
PMU counter="cube_busy_cycles"  ratio=0.059215 … value=2746656.0  evidence=measured
PMU counter="pmu_total_cycles"  ratio=1.0      … evidence=measured
```

解读：qk_pv 的瓶颈画像是 **scalar 40% + mte2 25%**——标量/地址计算和 GM→片上搬运主导，cube 只占 6%。这决定了优化方向（减少标量开销与访存），也让"该换 tiling 还是该减指令"这类讨论有了共同证据。`ratio = value/total_cycles`，无 total 列时会显式 `unavailable` 并说明原因。

## 8b. 其余模态（查 run 2）

```bash
pfdb query args_dump   --run-id 2 | head -3
pfdb query scope_stats --run-id 2
pfdb query incore      --run-id 2
pfdb query memory      --run-id 2
pfdb query bench       --run-id 2
```

```text
ARGS arg_index=4 bin_size=0 dtype="BFLOAT16" kind="tensor" role="input" run_id=2 seq=0 shape=[512,64] stage="before_dispatch" task_id="4294967303" task_id_raw="0x0000000100000007" task_id_u64="4294967303" evidence=measured
SCOPE payload={"fatal":false,"dropped":0,"total":154,…} run_id=2 seq=0 evidence=measured
SCOPE payload={"depth":2,…} phase="begin" ring=2 run_id=2 seq=3 site="decode_csa_tp1_test.cpp:94" evidence=measured
INCORE run_id=2 evidence=unavailable
MEMORY run_id=2 evidence=unavailable
BENCH max_us=3944.2 mean_us=3686.013 median_us=3663.05 min_us=3534.8 rounds=300 run_id=2 evidence=measured
```

缺的模态（本演示没跑 in-core 仿真、编译报告无 memory dump）返回 `unavailable` 而不是编造；`inventory` 会解释每个模态的 `requested/state`。args_dump 行同时保留 `task_id_raw`（十六进制）与 `task_id_u64`（十进制），方便与 PMU/泳道对账。

---

## 9. 可视化

```bash
pfdb render whole --run 1 --render-dir .pfdb/demo-render
pfdb render window --run 1 --t0 1830 --t1 1900 --render-dir .pfdb/demo-render
pfdb render task  --run 1 --task-id 8589934646 --render-dir .pfdb/demo-render
pfdb render core  --run 1 --core 0 --render-dir .pfdb/demo-render
```

每次返回一个 `IMAGE` 事实（路径、sha256、尺寸、µs/px、图例）+ PNG 文件：

```text
IMAGE cache_hit=false downsampled=false height=450 kind="whole" legend={"aic":"#1f77b4","aiv":"#ff7f0e"} path=".pfdb/demo-render/1/whole-6affb8b1b1c6be3b.png" run_id=1 sha256="803b…" size_bytes=68622 us_per_px=4.97605 width=800 x0_us=75.72 x1_us=4056.56 evidence=measured
```

- **whole**：60 核 × 4ms 全景——AIC（蓝，0–19）与 AIV（橙，20–59）的计算波次、串行段、drain 尾一张图看全；给不熟悉数据的人看，这一张就够了；
- **window**：任意时间窗放大，附依赖箭头（上限 200 条）；
- **task**：目标任务红色高亮 + 生产者/消费者淡化 + 绿色就绪线；
- **core**：单核时间轴，空闲段黄色着色。

同参数重复渲染逐字节一致（确定性缓存）；`.manifest.json` 记录 sha256 与图例。多模态 agent 可直接走 MCP 的 `pfdb.render` 拿到 base64 PNG。

> 缓存键包含 run 数据指纹（records sha256），因此即使多个数据库共享同一渲染目录、或删库重建后 run_id 复用，也不会命中另一个采集的旧图。

---

## 10. 调优闭环：note → baseline → compare → trial

```bash
pfdb note 2 "run B carries pmu/scope_stats/dump_args; timing perturbed by observers"
pfdb baseline add 1 --name csa-baseline --bench-mean 3686.0
pfdb compare 1 2 --family --bootstrap --resamples 2000
```

真实输出（节选）：

```text
COMPARE compatible=true program="decode_csa_tp1" run_a=1 run_b=2 evidence=measured
NOTE bench_mean_us="unprofiled" makespan_us="profiled-with-observer" topic="metric-scope" evidence=proven
DELTA after=3686.013 before=3685.488 delta=0.525 metric="bench_mean_us" ratio=1.0 run_a=1 run_b=2 evidence=measured
DELTA after=766.68 before=798.38 delta=-31.7 family="qk_pv" metric="family_busy_us" ratio=0.96 run_a=1 run_b=2 evidence=measured
DELTA after=72.0 before=72.0 delta=0.0 metric="tasks" ratio=1.0 run_a=1 run_b=2 evidence=measured
…
CONFIDENCE baseline_mean_us=3685.488 candidate_mean_us=3686.013 ci_high=0.003 ci_low=-0.003 confidence=0.95 metric="bench_mean_speedup" resamples=2000.0 samples_per_stratum=[100,100,100] seed=0.0 speedup=-0.0 strata=3.0 evidence=measured
```

解读（也是给观众的关键一课）：

- 两次采集是**同一段代码**，bench 差 0.5µs、置信区间 [−0.3%, +0.3%] 覆盖 0——真实性能相同；
- 而家族层面 `qk_pv` busy 差了 −31.7µs（run-to-run 波动）；makespan 层面差异巨大（3984 vs 72813µs）——**全是观测开销**；
- `NOTE` 事实持续提醒口径：裁决看 bench，解释看泳道。

记录裁决（数据库不替你判断，verdict 是你决定后的记录动作）：

```bash
pfdb trial register --goal "establish decode_csa tp1 baseline evidence" \
    --hypothesis "run B modalities perturb makespan but not unprofiled bench"
pfdb trial bind 1 2
pfdb trial verdict 1 --verdict neutral \
    --evidence "bench CI covers 0; makespan delta is observer overhead"
pfdb trial list
```

```text
trial 1 registered
trial 1 bound to run 2
trial 1 verdict=neutral
TRIAL bench_mean_us=3686.0133333333333 changed_files=[] evidence_refs=["bench CI covers 0; makespan delta is observer overhead"] goal="establish decode_csa tp1 baseline evidence" … run_id=2 status="done" trial_id=1 verdict="neutral" evidence=measured
```

`baseline` 同时保护 run 1 不被 prune；真正的改代码战役里，同一套命令就是完整的调优记录（goal/hypothesis/changed_files/血缘/裁决证据全可回溯）。

---

## 11. Agent 通道：MCP 会话

pfdb 的 38 个工具可以原样暴露给 MCP 客户端（Claude/Cursor/自研 agent）。仓库自带的 mock agent 只用 MCP 工具走完一次 Z0→Z4 完整会话：

```bash
PFDB_PATH=$PWD/.pfdb/demo/profile.duckdb python profile_db/examples/mock_agent.py
```

真实输出（节选）：

```text
# 38 tools: pfdb.list_runs, pfdb.overview, pfdb.inventory, pfdb.density, …, pfdb.render, pfdb.version
== list_runs ==
RUN bench_mean_us=3685.488 … program="decode_csa_tp1" … run_id=1 tasks=72 evidence=measured
== overview ==
METRIC … cpm_us=3753.3 … makespan_us=3984.04 … tasks=72 time_bands=1594 evidence=measured
== density ==
BAND band_idx=0 busy_cores=20 …
== why_sparse == … == region == … == task == … == deps == … == why_late ==
STALL run_id=1 task_id="12884901889" upstream_depth=0 evidence=unavailable
# session complete (run_id=1)
```

把 `pfdb serve --mcp` 作为子进程挂给你的 agent 即可复现同款工具面（只读加 `--writable` 才能写 trial/note）。

---

## 12. 复现注意事项

1. **`--tp 1` 必须显式传**：默认 `--tp 2` 要两张卡；
2. **`--enable-chip-swimlane 4` 要显式写数字**：这些入口的裸 flag 等于 level 1，会让 makespan/关键路径全部 `unavailable`；
3. **时间戳目录定位**：每次采集落在新的 `build_output/_jit_<case>_<hash>/`，用 `grep "Swimlane JSON written" <日志>` 或 `ls -t build_output | head` 找到本次目录，再用 shell 变量传递；
4. **数值漂移**：makespan/bench/task_id 都会随设备状态与图版本变化，演示时讲结构、以实跑输出为准；
5. **run B 的 makespan 天然巨大**：全模态采集的观测开销（本演示 18 倍），这是教学点不是故障；
6. **删库即重置**：`rm -rf .pfdb/demo` 后重新 `pfdb init && pfdb ingest …` 即可从 `build_output/` 完整重建——数据库是可弃工作集；
7. **单写者**：ingest/写 trial 与 `serve --writable` 不要并行；遇到 `LockError` 等另一个写者结束即可；
8. **真实改码战役**：把本演示的 compare/trial 一节套在"改动前后各采集一次（各自 3 份 bench）"上，就是完整的调优记录闭环。

---

## 13. 相关文档

- [pfdb 使用手册（全功能参考）](./usage-manual.zh.md)
- [profile_db/README.md](../README.md) 与 [profile_db/DESIGN.md](../DESIGN.md)
- 采集命令全景（英文）：`docs/debug-and-tune/profiling-options.md`
- agent 用法速查（英文）：`.agents/skills/profile-feedback/SKILL.md`
