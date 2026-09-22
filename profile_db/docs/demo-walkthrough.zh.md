# pfdb 实机演示：DeepSeek-V4 CSA Decode 算子全流程

本演示用仓库中的真实大算子 `models/deepseek_v4_flash_dspark/decode_csa.py`（DeepSeek-V4 上下文并行 CSA decode 层，TP=1 单卡形态）走完 pfdb 的全部核心环节：**真机采集 → 入库 → 宏观体检 → 关键路径定位 → 单任务微归因 → 管线画像 → 可视化 → 调优闭环 → MCP agent 会话**。

下面的命令可以按顺序复制。样例输出来自某次实机（仅对含本机绝对路径的字段做截断脱敏），用来说明「这一步看起来像什么」；你自己的数字、`task_id`、空带位置都会漂，**以你终端里的输出为准，跟着结构走**。

> 配套文档：各命令/查询的完整参数与语义见[使用手册](./usage-manual.zh.md)。

---

## 0. 演示亮点一览

| 环节 | 展示的能力 | 本次实测（会漂） |
|---|---|---|
| 入库 | 大图一次 ingest：约 72 任务 / 4271 物理行 / 242 依赖边 / 219 条编译器提示。干净泳道约 14 s；带 PMU 的 run 约 2 min | §4 |
| 宏观体检 | 两条引擎 60 核的分带占用、稀疏带确定性归因 | §5 |
| 关键路径 | observed CPM 二十余个任务；CLI 按路径顺序，stall 需自行排序 | §6 |
| 微归因 | FIN→dispatch→receive→start 四段分解、时间戳证明的早派发、调度相流 | §7 |
| 管线画像 | qk_pv 融合核的 PMU：scalar ~40% / mte2 ~25% / cube ~6% | §8 |
| 可视化 | 60 核 × ~4 ms 全景泳道 PNG | §9 |
| 调优闭环 | compare + 分层 bootstrap 置信区间 + trial 裁决记录 | §10 |
| Agent 通道 | 38 个 MCP 工具的完整会话 | §11 |
| 观测开销 | 干净采集 vs 全模态采集的 makespan 差一个数量级——裁决必须看 bench | §10 |

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 环境 | conda `pypto`（PyPTO 运行时 + pfdb）。`pfdb --version` 应为 `0.3.1` 或更新 |
| 设备 | 一张**空闲** 910B（`-p a2a3`）。用 `npu-smi info` 看 AICore% 接近 0、无占用进程的卡，记下 id |
| 算子 | `models/deepseek_v4_flash_dspark/decode_csa.py`，暴露 `--tp {1,2,4}`、`--enable-chip-swimlane [{0..4}]`、`--enable-pmu`、`--enable-dump-args`、`--enable-scope-stats`、`--save-data` / `--golden-data` |
| 墙钟 | 第一次运行约 3 分钟，其中 **golden 参考约占 2 分钟**，compile 有缓存时只有数秒。后面用 `--golden-data` 跳过 golden，每次约 20–30 s（bench 100 轮）到约 3 分钟（带泳道） |

> **为什么 `--tp 1`**：脚本默认 `--tp 2` 需要 2 张卡（`-d 0,1`）。单卡演示必须显式 `--tp 1`。
>
> **全程同一个 shell**，并且每次新开终端都要重新 `conda activate pypto` 和 `export PFDB_PATH=…`。

---

## 2. 准备工作区

`pfdb init` **幂等、不会清空已有库**。重跑本演示请先删目录。

```bash
conda activate pypto
cd <仓库根目录>
export PYTHONPATH="$PWD"

# 空闲卡 id（下面所有 -d / --device 必须一致）
export DEV=0
npu-smi info   # 确认 $DEV 空闲；被占就改成别的卡

# 演示专属数据库（.pfdb/ 已 gitignore）
export PFDB_PATH=$PWD/.pfdb/demo/profile.duckdb
rm -rf .pfdb/demo
mkdir -p .pfdb/demo
pfdb init
```

输出：

```text
pfdb initialized at …/.pfdb/demo/profile.duckdb (schema_version=7)
```

> **每个 shell 都要 `export PFDB_PATH`**。忘了就会写到默认路径，之后查出来是「空库」。

---

## 3. 采集（pfdb 之外，但决定了 pfdb 有什么数据）

pfdb 自己不采集。本演示需要 5 次真机运行：1 次干净泳道、1 次全模态、3 次无观测 bench。

第一次带 `--save-data`，把 golden 快照留下来；后面三次 bench 和 Run B 用 `--golden-data` 跳过约 130 s 的 torch 参考。这不改变内核数值，只跳过 host 侧重算。

### 3.1 Run A：干净计时采集（level-4 泳道）

```bash
python models/deepseek_v4_flash_dspark/decode_csa.py -p a2a3 -d "$DEV" --tp 1 \
    --enable-chip-swimlane 4 --save-data 2>&1 | tee capture_A.log
```

要点输出：

```text
[swimlane] chip swimlane enabled -> running the kernel twice (dep_gen perturbs timing, …):
[swimlane] run 1/2: capturing the task dependency graph (deps.json) in a subprocess; …
[swimlane] run 2/2: measuring clean per-task timing (this run's numbers are the ones reported).
Swimlane JSON written to: build_output/_jit_decode_csa_tp1_test_<hashA>/dfx_outputs/merged_swimlane_<ts>.json
[RUN] PASS (167.20s)
```

运行时自动跑两遍：第一遍子进程抓依赖图，第二遍干净计时。从日志抽出目录（不要手抄文档里的旧 hash）：

```bash
export HASH_A=$(grep -oE '_jit_decode_csa_tp1_test_[A-Za-z0-9_]+' capture_A.log | head -1)
export GOLDEN="$PWD/build_output/${HASH_A}/data"
echo "HASH_A=$HASH_A"
ls "$GOLDEN/in" "$GOLDEN/out" >/dev/null && echo "golden snapshot ok"
```

### 3.2 Run B：全模态采集（swimlane + PMU + 作用域统计 + 参数转储）

```bash
python models/deepseek_v4_flash_dspark/decode_csa.py -p a2a3 -d "$DEV" --tp 1 \
    --enable-chip-swimlane 4 --enable-pmu 2 --enable-scope-stats --enable-dump-args 2 \
    --golden-data "$GOLDEN" \
    2>&1 | tee capture_B.log

export HASH_B=$(grep -oE '_jit_decode_csa_tp1_test_[A-Za-z0-9_]+' capture_B.log | head -1)
echo "HASH_B=$HASH_B"
ls "build_output/${HASH_B}/dfx_outputs/"
```

期望看到：

```text
args_dump  chip_swimlane_records.json  deps.json  merged_swimlane_<ts>.json
name_map__jit_decode_csa_tp1_test_<hashB>.json  pmu.csv  profile_capture_manifest.json  scope_stats
```

### 3.3 三份独立无观测基准（裁决数字）

三次必须是**三个独立进程**。`PYPTO_BENCH_RAW=1` 让日志带上逐轮原始样本（`headline raw n=100 eff_us=[…]`），后面的分层 bootstrap 两边都要这三份 log。

```bash
for i in 1 2 3; do
  PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 \
    python models/deepseek_v4_flash_dspark/decode_csa.py -p a2a3 -d "$DEV" --tp 1 \
    --golden-data "$GOLDEN" \
    > bench_run$i.log 2>&1
  grep -E 'PASS|FAIL|effective_us' bench_run$i.log | tail -3
done
```

某次实测（三次独立调用，每次 100 轮）：

```text
[RUN]   effective_us (100 rounds) min=3534.8 median=3648.2 mean=3668.7 max=3885.6
[RUN]   effective_us (100 rounds) min=3545.8 median=3663.1 mean=3692.2 max=3944.2
[RUN]   effective_us (100 rounds) min=3545.6 median=3677.6 mean=3697.2 max=3888.0
[RUN] PASS
```

每份 log 里都应有 `headline raw n=100`。没有这一行，ingest 进得去，但 §10 的 `--bootstrap` 会失败。

---

## 4. 入库

**两 run 都要挂上同一组 `--bench-log`。** 本演示比较的是「同一段代码、不同观测器」，无观测 bench 只有一份；两边各挂 ≥3 层 raw 样本，`compare --bootstrap` 才不会报 `stratified bootstrap requires at least three raw benchmark strata`。

`--device` 填采集时的 `$DEV`，不是写死的 0。

```bash
# Run A：干净采集 + 3 份独立 bench 分层
pfdb ingest "build_output/${HASH_A}/dfx_outputs" \
    --program decode_csa_tp1 --platform a2a3 --device "$DEV" --no-prune \
    --bench-log bench_run1.log --bench-log bench_run2.log --bench-log bench_run3.log \
    --notes "decode_csa run A: level-4 swimlane + 3 independent bench strata" --tags demo

# Run B：全模态采集 + 同一组无观测 bench（代码没变，只多了观测器）
pfdb ingest "build_output/${HASH_B}/dfx_outputs" \
    --program decode_csa_tp1 --platform a2a3 --device "$DEV" --no-prune \
    --bench-log bench_run1.log --bench-log bench_run2.log --bench-log bench_run3.log \
    --notes "decode_csa run B: swimlane + pmu + scope_stats + dump_args" --tags demo

pfdb list
```

某次实测：

```text
ingested run 1 (program=decode_csa_tp1 level=4 mode=link)
  tasks=72 task_rows=4271 edges=242 artifacts=5 perf_hints=219 memory=0 pmu=0 makespan=3984.040us
  ingest_wall_ms=13386.781 …
ingested run 2 (program=decode_csa_tp1 level=4 mode=link)
  tasks=72 task_rows=4271 edges=242 artifacts=8 perf_hints=219 memory=0 pmu=38439 makespan=72813.520us
  ingest_wall_ms=112442.024 …
RUN bench_mean_us=3685.488 … makespan_us=3984.04 … run_id=1 tasks=72 evidence=measured
RUN bench_mean_us=3685.488 … makespan_us=72813.52 … run_id=2 tasks=72 evidence=measured
```

检查清单：`run_id` 为 1 和 2、`level=4`、`program="decode_csa_tp1"`、两边都有 `bench_mean_us`、run 2 的 makespan 远大于 run 1。若库里已有别的 run，用 `pfdb list` 的 id，不要假设一定是 1/2。下文用 1=干净泳道、2=全模态。

这一步展示了什么：

- **大图入库**：4271 行物理时序 + 242 条边 + 衍生层（密度带、空闲段、双关键路径）在一个事务里落表。干净泳道大约十几秒；run B 解析数万条 PMU，ingest 约 2 分钟，属正常；
- **`--program` 归一化**：两次采集的 `_jit_<hash>` 目录名不同，统一传 `--program decode_csa_tp1` 才能让后面的 compare 认它们是同一程序；
- **run 2 的 makespan 大一个数量级**不是代码变慢，是 pmu / dump_args / scope_stats 的观测开销。**时序结论看 run 1，模态内容查 run 2**；两边 bench 相同（同一组 log）或几乎相同，证明真实性能没变；
- 幂等：同一 `dfx_outputs` 再 ingest 一次，行数不变。

把后面要用的 id 抽出来（不要手抄文档里的旧 id）：

```bash
export QK_PV=$(pfdb query tasks --run-id 1 --family qk_pv --format json \
  | python -c "import json,sys; print(json.load(sys.stdin)[0]['task_id'])")
echo "QK_PV=$QK_PV"
```

---

## 5. 宏观体检（Z0/Z1）

### 5.1 顶线指标

```bash
pfdb query overview --run-id 1
```

```text
RUN clock_freq_hz=50000000 … level=4 num_cores=60 platform="a2a3" program="decode_csa_tp1" … run_id=1 evidence=measured
METRIC artifacts=5 bench_rounds=300 cpm_us=3753.3 edges=242 idle_gaps=2900 makespan_us=3984.04 … tasks=72 time_bands=1594 evidence=measured
RESOURCE cores=20 engine="aic" run_id=1 evidence=measured
RESOURCE cores=40 engine="aiv" run_id=1 evidence=measured
```

解读：60 核（20 AIC + 40 AIV）；本次 CPM 约占 makespan 的 **94%**——调度已经相当紧凑，优化空间主要在关键路径本身而非调度空洞。先确认 `level=4`，否则 §7 的 FIN/dispatch 查询会 `unavailable`。

### 5.2 密度带

```bash
pfdb query density --run-id 1 --bands 20
```

```text
BAND band_idx=0 busy_cores=20 … engine="aic" sparse=false t0_us=75.72 t1_us=275.72 …
BAND band_idx=1 busy_cores=0 … engine="aic" sparse=true …
…
BAND band_idx=19 busy_cores=0 drain_tail=true engine="aic" …
TRUNCATED first_dropped_index=20 remaining=20 limit=4096 hint="retry --budget 32768"
```

解读：AIC 多数带满载（20/20）；哪一条是空带会随运行变化。默认预算只打印 AIC 20 条，AIV 被 TRUNCATED，`hint` 教你加 `--budget`。空带出现在不同 `band_idx` 是正常的。

### 5.3 稀疏带归因

```bash
pfdb query sparse_regions --run-id 1 --top-k 3
```

```text
SPARSE busy_cores=0 engine="aic" kind="ready_starved" lagging_producer="8589934603" … stored_band_idx=26 … evidence=proven
SPARSE … lagging_producer="8589934603" … stored_band_idx=27 … evidence=proven
SPARSE … lagging_producer="8589934603" … stored_band_idx=28 … evidence=proven
```

`kind="ready_starved"` 且 `evidence=proven` 表示：由记录的确定性结构推出「上游没喂上」，不是猜测。接着查那个生产者是谁（可能是真实算子，不要预先当成 dummy）：

```bash
export LAG=$(pfdb query sparse_regions --run-id 1 --top-k 1 --format json \
  | python -c "import json,sys; print(json.load(sys.stdin)[0]['lagging_producer'])")
pfdb query task --run-id 1 --task-id "$LAG"
```

`stored_band_idx` 是存储带坐标。若要问「为什么这条带空」，用 `pfdb query why_sparse --run-id 1 --stored-band <idx>`，不要和 `density --bands N` 的 `band_idx` 混用。

---

## 6. 关键路径（Z4）

默认按路径 **seq** 输出，**不是**按 stall 排序：

```bash
pfdb query critical_path --run-id 1 --kind observed
```

PATH 行自带 `name` / `family` / `engine` / `early_dispatch_proven` / `gap_kind`。某次路径上 `qk_pv` wall 约占 makespan 的 20%，是最大时间贡献者；单步 stall 通常只有数 µs 到十几 µs。`--kind static` 可以交叉验证静态最长路径。

按 stall 看前几名（CLI 没有 `--sort`，用 json）：

```bash
pfdb query critical_path --run-id 1 --kind observed --format json --budget 32768 \
  | python -c "
import json, sys
rows = [r for r in json.load(sys.stdin) if r.get('rec') == 'PATH']
rows.sort(key=lambda r: -(r.get('stall_us') or 0))
for r in rows[:8]:
    print(f\"stall={r.get('stall_us')} wall={r.get('wall_us')} family={r.get('family')} gap_kind={r.get('gap_kind')} early={r.get('early_dispatch_proven')} task_id={r.get('task_id')}\")
"
```

某次 stall 排序节选（task_id / 名次会变）：

```text
stall=14.5  wall=213.32 family=idx_qr_proj_matmul  gap_kind=data-wait early=…
stall=9.1   wall=799.7  family=qk_pv               gap_kind=data-wait …
```

记下一条 `early_dispatch_proven` 为 `full` 或 `partial` 的 `task_id` 供 §7.2 使用。若本次路径上全是 `none`，用下面命令在全图里找一条 proven 的：

```bash
export EARLY_TASK=$(pfdb query critical_path --run-id 1 --kind observed --format json --budget 32768 \
  | python -c "
import json, sys
rows = [r for r in json.load(sys.stdin) if r.get('rec')=='PATH' and r.get('early_dispatch_proven') in ('full','partial')]
print(rows[0]['task_id'] if rows else '')
")
# 路径上没有 proven 时，换一个已知会早派发的短任务再查（family 名以你的 PATH 为准）
if [ -z "$EARLY_TASK" ]; then
  export EARLY_TASK=$(pfdb query tasks --run-id 1 --family hc_pre_linear_reduce --format json \
    | python -c "import json,sys; print(json.load(sys.stdin)[0]['task_id'])")
fi
echo "EARLY_TASK=$EARLY_TASK"
```

---

## 7. 单任务微归因（Z4 下钻）

### 7.1 最大耗时任务：qk_pv

```bash
pfdb query task --run-id 1 --task-id "$QK_PV"
```

```text
TASK block_num=24 busy_us=798.38 early_dispatch_flag=true engine="mixed" family="qk_pv" … num_rows=72 on_cpm_observed=true … wall_us=799.7 evidence=measured
```

72 个物理行铺在 AIC+AIV 两类核上（`engine="mixed"`）。记下 `min_dispatch_us`，§9 的窗口用它：

```bash
export QK_T0=$(pfdb query task --run-id 1 --task-id "$QK_PV" --format json \
  | python -c "import json,sys; print(json.load(sys.stdin)[0]['min_dispatch_us'])")
export QK_T1=$(python -c "print(float('$QK_T0') + 70)")
echo "window $QK_T0 .. $QK_T1"

pfdb query rows     --run-id 1 --task-id "$QK_PV" | head -3
pfdb query why_long --run-id 1 --task-id "$QK_PV"
pfdb query deps     --run-id 1 --task-id "$QK_PV" --direction in
```

某次实测：行级 busy 约 353–444 µs（核间散差）；`deps --direction in` 会同时看到 `explicit` / `creator` / `tensormap` 三种边。

### 7.2 早派发被时间戳证明

不要假设「stall 最大的那个任务一定早派发了」。用 §6 选出的 `$EARLY_TASK`：

```bash
pfdb query why_late       --run-id 1 --task-id "$EARLY_TASK"
pfdb query early_dispatch --run-id 1 --task-id "$EARLY_TASK"
```

某次 `data-wait` 短任务的样子：

```text
STALL dispatch_us=846.92 dispatch_wait_us=2.16 fin_detect_us=-2.12 gap_us=1.22 ready_us=849.04 … start_wait_us=1.18 … upstream_depth=6 evidence=measured
EARLY proven_blocks=7 ready_us=849.04 status="partial" tol_us=0.04 total_blocks=24 evidence=proven
```

- `why_late` 恒等式：`gap_us = fin_detect_us + dispatch_wait_us + start_wait_us`（`fin_detect_us` 为负表示 FIN 晚于 dispatch，即提前派发）；
- `early_dispatch` 四态：`full` / `partial` / `none` / `unavailable`。`status="partial"` 且 `evidence=proven` 表示部分块的时间戳证明提前派发（容差 2 拍 = 0.04 µs）。代码里的 `allow_early_resolve` 和硬件上是否真的提前派发是两回事；
- 若某次选中的任务是 `core-wait`，`early_dispatch` 常为 `none`（`evidence=unproven`）。换一条 `early_dispatch_proven=full|partial` 的即可，不是 pfdb 坏了。

### 7.3 调度相流

```bash
pfdb query scheduler --run-id 1 --task-id "$EARLY_TASK"
```

```text
SCHED kind="dispatch" lane=0 … 
SCHED kind="early_dispatch" lane=0 … tasks_processed=7 …
…
TRUNCATED first_dropped_index=35 remaining=50 limit=4096 hint="retry --budget 32768"
```

任务窗口（±1 µs）会叠很多调度相。`--budget 32768` 可看全。相条数随任务而变。

---

## 8. 管线画像：PMU（查 run 2）

PMU 只在全模态采集里。`task_id` 与 run 1 的 `$QK_PV` 通常相同（同一张图）：

```bash
pfdb query pmu --run-id 2 --task-id "$QK_PV"
```

```text
MODALITY modality="pmu" … state="available" evidence=measured
PMU_SUMMARY counters=9 measurements=648 samples=72 …
PMU counter="scalar_busy_cycles" ratio=0.402731 …
PMU counter="mte2_busy_cycles"  ratio=0.2504 …
PMU counter="vec_busy_cycles"   ratio=0.196556 …
PMU counter="cube_busy_cycles"  ratio=0.059215 …
PMU counter="pmu_total_cycles"  ratio=1.0 …
```

解读：qk_pv 的画像是 **scalar ~40% + mte2 ~25%**，cube 约 6%——标量/地址计算和 GM→片上搬运主导。`ratio = value/total_cycles`；没有 total 列时会显式 `unavailable`。

## 8b. 其余模态（查 run 2）

```bash
pfdb query args_dump   --run-id 2 | head -3
pfdb query scope_stats --run-id 2
pfdb query incore      --run-id 2
pfdb query memory      --run-id 2
pfdb query bench       --run-id 1 | head -5
pfdb query bench       --run-id 2 | head -5
pfdb query inventory   --run-id 2
```

```text
ARGS … stage="before_dispatch" task_id="…" task_id_raw="0x…" task_id_u64="…" evidence=measured
SCOPE payload={…} run_id=2 seq=0 evidence=measured
INCORE run_id=2 evidence=unavailable
MEMORY run_id=2 evidence=unavailable
BENCH … rounds=300 run_id=1 evidence=measured
BENCH … rounds=300 run_id=2 evidence=measured
```

本演示没跑 in-core、也没有 memory dump，这两种返回 `unavailable` 是预期。`inventory` 解释每个模态的 `requested` / `state`。args_dump 同时保留十六进制 `task_id_raw` 和十进制 `task_id_u64`。

若 `query bench --run-id 2` 为 unavailable，回头检查 §4 是否给 B 也传了三份 `--bench-log`。

---

## 9. 可视化

窗口必须用你自己的 `$QK_T0` / `$QK_T1`（§7.1），不要复制别人捕获里的 1830–1900。

```bash
pfdb render whole --run 1 --render-dir .pfdb/demo-render
pfdb render window --run 1 --t0 "$QK_T0" --t1 "$QK_T1" --render-dir .pfdb/demo-render
pfdb render task  --run 1 --task-id "$QK_PV" --render-dir .pfdb/demo-render
pfdb render core  --run 1 --core 0 --render-dir .pfdb/demo-render
```

每次返回一个 `IMAGE` 事实（路径、sha256、尺寸、µs/px、图例）+ PNG：

```text
IMAGE cache_hit=false kind="whole" path=".pfdb/demo-render/1/whole-….png" run_id=1 … width=800 evidence=measured
```

- **whole**：60 核 × ~4 ms 全景——AIC（蓝，0–19）与 AIV（橙，20–59）；
- **window**：任意时间窗放大，附依赖箭头（上限 200 条）；
- **task**：目标任务红色高亮 + 生产者/消费者淡化 + 绿色就绪线；
- **core**：单核时间轴，空闲段黄色着色。

同参数重复渲染逐字节一致。缓存键含 run 数据指纹，多库共享渲染目录或删库后 run_id 复用也不会串图。

---

## 10. 调优闭环：note → baseline → compare → trial

`--bench-mean` 可以省略，默认用 run 上已入库的 bench。两边都挂了同一组 raw log 时，`bench_mean_us` 的 delta 为 0，CI 覆盖 0；makespan 仍然差一个数量级。这正是本演示要证明的。

```bash
pfdb note 2 "run B carries pmu/scope_stats/dump_args; timing perturbed by observers"
pfdb baseline add 1 --name csa-baseline
pfdb compare 1 2 --family --bootstrap --resamples 2000
```

若报 `stratified bootstrap requires at least three raw benchmark strata`，对缺 log 的那一侧再 ingest 一次并带上三份 `--bench-log`（幂等，不会另开一个 run）。

某次实测（节选）：

```text
COMPARE compatible=true program="decode_csa_tp1" run_a=1 run_b=2 evidence=measured
NOTE bench_mean_us="unprofiled" makespan_us="profiled-with-observer" topic="metric-scope" evidence=proven
DELTA after=3685.488 before=3685.488 delta=0.0 metric="bench_mean_us" ratio=1.0 …
DELTA after=72813.52 before=3984.04  delta=68829.48 metric="makespan_us" ratio=18.277 …
DELTA after=766.68 before=798.38 delta=-31.7 family="qk_pv" metric="family_busy_us" …
CONFIDENCE … ci_high=0.004 ci_low=-0.004 confidence=0.95 metric="bench_mean_speedup" … strata=3.0 evidence=measured
```

解读：

- 两次采集是**同一段代码**；bench CI 覆盖 0——真实性能相同；
- 家族 busy 仍会有数十 µs 的 run-to-run 波动；makespan 的巨大差异**全是观测开销**；
- `NOTE` 提醒口径：裁决看 bench，解释看泳道。

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
TRIAL … run_id=2 status="done" trial_id=1 verdict="neutral" evidence=measured
```

`bind 1 2` 把 trial 绑到**第二个** run（候选，这里是全模态的 run 2）。`baseline` 保护 run 1 不被 prune。真正改代码时：改动前后各采一次（各自 3 份 bench），同一套命令就是完整记录。

---

## 11. Agent 通道：MCP 会话

```bash
PFDB_PATH=$PWD/.pfdb/demo/profile.duckdb python profile_db/examples/mock_agent.py
```

```text
# 38 tools: pfdb.list_runs, pfdb.overview, …, pfdb.render, pfdb.version
== list_runs ==
RUN … run_id=1 …
== overview ==
METRIC … makespan_us=… …
== density ==
BAND band_idx=0 …
== why_sparse == … == region == … == task == … == deps == … == why_late ==
STALL run_id=1 task_id="12884901889" upstream_depth=0 evidence=unavailable
# session complete (run_id=1)
```

mock agent 从 `region` 里取**第一个** `task_id`（常常是路径前端的 seed 类任务）。这类任务没有上游 stall，`why_late` 返回 `unavailable` **是脚本的取任务方式，不是 MCP 失败**。会话以 `# session complete` 结束即成功。

把 `pfdb serve --mcp` 作为子进程挂给你的 agent 即可复现同款工具面（写 trial/note 需要 `--writable`）。

---

## 12. 复现注意事项

1. **`--tp 1` 必须显式传**：默认 `--tp 2` 要两张卡；
2. **`--enable-chip-swimlane 4` 要显式写数字**：裸 flag 等于 level 1，makespan / 关键路径会全部 `unavailable`；
3. **卡号一致**：`-d "$DEV"` 与 ingest `--device "$DEV"` 相同；0/1 被占就换空闲卡；
4. **用日志里的 hash，不要抄文档**：`HASH_A` / `HASH_B` 来自 `grep`；`QK_PV` / `EARLY_TASK` 来自 `pfdb query … --format json`；
5. **两 run 都要 3 份 `--bench-log`**：否则 `query bench --run-id 2` 为空、`compare --bootstrap` 报错。本演示代码没变，挂同一组 log 是对的；
6. **数值漂移**：makespan / bench / 空带位置 / 谁被早派发都会变。讲结构，以实跑为准；
7. **run B 的 makespan 天然巨大**：全模态观测开销（约十几倍），这是教学点不是故障；
8. **`pfdb init` 不清库**：重跑先 `rm -rf .pfdb/demo`；
9. **单写者**：ingest / 写 trial 与 `serve --writable` 不要并行；遇到 `LockError` 等另一个写者结束；
10. **golden 快照**：`data/in` 与 `data/out` 都在才能 `--golden-data`。规格或参考实现变了要重新 `--save-data`；
11. **真实改码战役**：改动前后各采集一次（各自 3 份独立 bench），再走 §10。

---

## 13. 相关文档

- [pfdb 使用手册（全功能参考）](./usage-manual.zh.md)
- [profile_db/README.md](../README.md) 与 [profile_db/DESIGN.md](../DESIGN.md)
- 采集命令全景（英文）：[`docs/debug-and-tune/profiling-options.md`](../../docs/debug-and-tune/profiling-options.md)
- golden 快照与 replay：[`docs/run-and-validate/save-and-replay.md`](../../docs/run-and-validate/save-and-replay.md)
- agent 用法速查（英文）：[`.claude/skills/profile-feedback/SKILL.md`](../../.claude/skills/profile-feedback/SKILL.md)
