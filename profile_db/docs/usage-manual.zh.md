# pfdb（profile_db）使用手册

> 版本基准：**profile-db 0.3.1**（数据库 schema v7 / MCP tool schema v3 / 渲染器 `profile_db.render/2`）。
> 本手册全部示例均在真实 Ascend 910B 设备（`a2a3`，device 0）上实跑验证，输出为原样摘录（仅对含本机绝对路径的字段做了截断脱敏）。数值会随每次运行漂移，复现时请以你自己的实跑输出为准。

本文是**完整参考手册**：覆盖安装、数据来源、入库、查询、渲染、生命周期管理、MCP 与 Python API 的全部功能。每个功能给出用途、参数、返回字段与经过实机验证的示例。如果你只想看一个端到端的完整故事（复杂大算子、从采集到调优结论），请阅读配套的[实机演示文档](./demo-walkthrough.zh.md)。

---

## 目录

1. [pfdb 是什么](#1-pfdb-是什么)
2. [安装与初始化](#2-安装与初始化)
3. [数据从哪里来（pfdb 不采集）](#3-数据从哪里来pfdb-不采集)
4. [入库：pfdb ingest 与 ingest-incore](#4-入库pfdb-ingest-与-ingest-incore)
5. [读懂输出：facts DSL 与证据状态](#5-读懂输出facts-dsl-与证据状态)
6. [查询体系：26 条查询逐条详解](#6-查询体系26-条查询逐条详解)
7. [可视化渲染：pfdb render](#7-可视化渲染pfdb-render)
8. [生命周期与短期记忆：prune / note / compare / baseline / trial](#8-生命周期与短期记忆)
9. [Agent 通道：MCP 服务](#9agent-通道mcp-服务)
10. [Python API](#10-python-api)
11. [数据库内部参考](#11-数据库内部参考)
12. [故障排查（全部实测）](#12-故障排查全部实测)
13. [附录：速查表](#13-附录速查表)

---

## 1. pfdb 是什么

pfdb 是面向调优工作流（人和 agent 皆可）的 **profiling 反馈数据库**。它把一次 PyPTO 采集（`build_output/<case>/dfx_outputs/` 下的泳道记录、依赖图、PMU 计数器等）转换成一个可查询的 DuckDB 工作集，并用**一条事实一行**的紧凑格式回答明确的调优问题。

三条不变的边界（"三不"原则）：

1. **不发起任何采集**：pfdb 只读取已经存在的工件，绝不运行内核、模拟器或设备命令；
2. **不修改任何源码**：调优 agent 在自己的职责下改代码，pfdb 只提供证据；
3. **不下优化结论**：查询只返回带证据状态的事实，`win`/`regression`/瓶颈标签必须由人（或调优 agent）自己判断后再记录回 trial 表。

设计要点：

- **查询优先、存储做薄**。profiling 数据强时效性，每轮修改后旧数据大多失去价值。默认 link 模式只记录工件路径 + sha256（不复制文件），整个 `.pfdb/` 目录随时可删、从 `build_output/` 重建。
- **分层查询**。Z0（定位）→ Z1（宏观密度）→ Z2（区域/核）→ Z3（单算子/依赖）→ Z4（微归因），每条查询回答一个有界问题。
- **证据状态**。每条事实恒带 `evidence=<measured|proven|unproven|unavailable>`，缺数据返回 `unavailable` 而不是猜测。
- **字节预算**。每条查询有默认输出预算，超预算显式 `TRUNCATED` 收尾并给出重试提示，永远不会静默截断。

### 1.1 术语表

| 术语 | 含义 |
|---|---|
| run | 一次采集入库后的整体，对应一份 `dfx_outputs`（按 records 文件 sha256 识别身份，幂等） |
| task | 图中一个调度任务（一次 dispatch）。`pl.spmd(n)` 是 1 个任务、n 个逻辑块 |
| task_row | 任务在一个物理核上的一行执行记录（一个任务可有多个物理行） |
| dep_edge | 依赖边，`pred` → `succ`，携带张量 dtype/shape/stride 元数据；`pred` 可以是 host 侧创建者伪节点 |
| engine | 执行引擎：`aic`（Cube）/ `aiv`（Vector）/ `mixed`（单任务跨两类核） |
| makespan_us | `max(finish) − min(dispatch)`，带 profiling 观测开销的调度跨度 |
| cpm_us | 关键路径（observed CPM）上各任务非重叠计算时间之和 |
| bench | `PYPTO_BENCH` 的**无观测**端到端计时，与 makespan 口径不同，永不互相比较 |
| fact | 查询输出的一行：`REC k=v ... evidence=<state>` |
| working set | 数据库当前保留的 run 集合（最新 K 个 + baseline/活跃 trial 引用的） |
| trial / baseline | 调优实验的短期记忆 / 命名的受保护基准 run |

---

## 2. 安装与初始化

### 2.1 安装

在 conda `pypto` 环境内：

```bash
pip install -e ./profile_db --no-build-isolation
pfdb --version
# pfdb 0.3.0
```

依赖：`duckdb`、`pydantic`、`matplotlib`（渲染）、`mcp`（仅 serve 需要），Python ≥ 3.10。若 `pfdb` 不在 PATH，可用等价形式：

```bash
PYTHONPATH=profile_db/src python -m profile_db --version
```

### 2.2 数据库路径解析顺序

`pfdb` 按以下优先级确定数据库文件（默认名 `profile.duckdb`）：

1. 显式 `--path`（仅 `pfdb init` 与 `pfdb serve` 支持该选项）；
2. 环境变量 `PFDB_PATH`；
3. `<当前目录>/.pfdb/profile.duckdb`。

> **最重要的习惯**：**在每个 shell 里都显式 `export PFDB_PATH=...`**。默认路径太容易被顺手写到，之后在另一个路径下查询会表现为"空库"，这是新手最常见的假故障。

### 2.3 初始化与目录布局

```bash
export PFDB_PATH=$PWD/.pfdb/profile.duckdb
pfdb init
# pfdb initialized at /…/.pfdb/profile.duckdb (schema_version=7)
```

`init` 幂等：已存在时只做迁移检查。运行时产生的文件布局：

```text
.pfdb/
├── profile.duckdb        # 数据库本体
├── profile.duckdb.lock   # 写者 flock 锁文件（存在即表示有人在写）
├── store/<run_id>/       # 仅 --copy 模式：工件归档副本
└── render/<run_id>/      # 渲染缓存（<db 所在目录>/render/）
```

`.pfdb/` 已加入 `.gitignore`，不入库。整个目录可随时删除后从 `build_output/` 重新 ingest 重建——这是设计允许的。

### 2.4 单写者模型

数据库是**单写者**：`ingest`、`prune`、`note`、`baseline add`、`trial ...` 以及 `serve --writable` 都持有排他写锁（`flock`，fail-fast）。第二个写者立即失败，不会排队：

```text
pfdb: error: database is locked by another DuckDB process: …/.pfdb/profile.duckdb;
wait for its write connection to close and retry
```

只读命令（`list`/`query`/`render`/`compare`/…）不加写锁，可与写者并存（DuckDB 层面的读写并发受其文件锁保护，冲突会以同样风格报错）。

---

## 3. 数据从哪里来（pfdb 不采集）

### 3.1 泳道采集：`--enable-chip-swimlane`

以任意 golden-harness 入口为例（示例入口与模型入口的 flag 拼写略有差异，**先看 `--help`**）：

```bash
# 示例入口：整数式 flag，裸 flag = level 1，完整采集必须显式写 4
python examples/beginner/hello_world.py -p a2a3 -d 0 --enable-chip-swimlane 4

# 模型入口：同为整数式
python models/deepseek_v4_flash_dspark/decode_csa.py -p a2a3 -d 0 --tp 1 --enable-chip-swimlane 4
```

采集级别与能力：

| Level | 采集内容 | pfdb 能回答的问题 |
|---|---|---|
| 1 | AICore（AIC/AIV）任务时间戳 | 任务/行级 timing；**无** AICPU FIN/dispatch 流 → makespan、critical_path、why_late 等报 `unavailable` |
| 2 | + AICPU 派发/完成时序 | makespan、关键路径、stall 分解 |
| 3 | + AICPU 调度相 | scheduler 查询有完整调度相数据 |
| 4 | + AICPU 编排相（完整采集） | 编排 submit 序、全部调度行为 |

真机泳道采集会自动跑两遍（第一遍在子进程里抓依赖图 `deps.json`，第二遍做干净计时）。注意两点：

- 部分入口的两遍采集的第一遍（dep_gen）可能失败（例如含步进标量的程序），此时 `deps.json` 缺失、**无法 ingest**——请检查采集日志中的 `RuntimeError: prepare_native_run failed` 与 `dfx_outputs/` 里是否存在 `deps.json`；
- 模拟器平台（`*sim`）的记录缺转换器所需的任务元数据，合并 trace 会跳过。

### 3.2 无观测基准：`PYPTO_BENCH`

```bash
PYPTO_BENCH=1 PYPTO_BENCH_RAW=1 python examples/beginner/hello_world.py -p a2a3 -d 0 \
  > bench_run1.log 2>&1
```

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `PYPTO_BENCH` | 关 | 打开计时循环（正确性验证之后） |
| `PYPTO_BENCH_ROUNDS` | 100 | 计时轮数 |
| `PYPTO_BENCH_WARMUP` | 5 | 预热轮（不入统计） |
| `PYPTO_BENCH_RAW` | 关 | 逐轮打印原始样本（`headline raw n=100 eff_us=[…]`） |

日志中的关键行（pfdb 的 `--bench-log` 依赖它）：

```text
[RUN]   effective_us (100 rounds) min=3534.8 median=3648.2 mean=3668.7 max=3885.6
[RUN]   headline raw n=100 eff_us=[3662.1, 3586.2, …]
```

调优结论以 bench 为准，泳道只负责解释变化；level-4 采集有观测开销，绝不用 makespan 当性能数字。

### 3.3 可选模态

| 模态 | 采集方式 | 产物 |
|---|---|---|
| PMU 计数器 | `--enable-pmu {0,1,2,4}`（部分入口经 config） | `dfx_outputs/pmu.csv` |
| 参数转储 | `--enable-dump-args {0..3}` | `dfx_outputs/args_dump/args_dump.json`（level 1 = 仅 `pl.dump_tag` 标记的张量带 payload；2 = 全任务 manifest、标记张量带 payload；3 = 全量最重） |
| 作用域统计 | `--enable-scope-stats` | `dfx_outputs/scope_stats/scope_stats.jsonl` |
| 依赖图 | `--enable-dep-gen`（泳道采集自动含此遍） | `dfx_outputs/deps.json` |
| in-core 仿真 | in-core profiling 流程另行生成 | `<case>/kernel_insight_all_funcs_*/manifest_export.csv` |
| 编译器提示 | 编译自动产生，无需 flag | `<case>/report/perf_hints.log` |

flag 暴露与否是**入口相关**的（`--help` 确认）；未暴露时可通过 harness 的 `config` dict 传入（如 `enable_pmu=2`）。哪些模态"被请求过"记录在 `dfx_outputs/profile_capture_manifest.json`，pfdb 据此区分 `not_requested` 与 `not_emitted`。

### 3.4 pfdb 读取的文件清单

`pfdb ingest <dir>` 中 `<dir>` 是 `dfx_outputs` 目录：

| 文件 | 必需？ | 说明 |
|---|---|---|
| `chip_swimlane_records.json` | **必需**（三者同名之一） | 现行名；兼容 `l2_swimlane_records.json` / `l2_perf_records.json` |
| `deps.json` | **必需** | 任务表 + 依赖边（`edges` 为空列表会被拒绝） |
| `name_map_*.json` | **必需**（恰一个） | callable_id → 内核名；program 名默认取自其文件名 |
| `pmu.csv` | 可选 | 自动发现，长表化入库 |
| `args_dump/args_dump.json` | 可选 | 自动发现，仅元数据（`args.bin` 永不入库） |
| `scope_stats/scope_stats.jsonl` | 可选 | 自动发现 |
| `../report/perf_hints.log` | 可选 | 在采集目录的父级 `report/` 下自动发现 |
| `merged_swimlane_*.json` | 可选 | 登记为工件，不解析 |

**新旧两代调度 schema 都支持**：现行格式（顶层 `scheduler_tasks{schema_version,producer,records}` + `scheduler_records{schema_version,streams[]}`，metrics 按 `record_index` 对齐）与归档格式（顶层 `aicpu_tasks` + 扁平 `aicpu_scheduler_phases`）自动识别，host 编排（`host_orchestrator_phases`，ns 时间戳）亦支持；两种编排时钟域同时出现会被拒绝。

---

## 4. 入库：pfdb ingest 与 ingest-incore

### 4.1 `pfdb ingest`

```bash
pfdb ingest <dfx_outputs> [--program P] [--platform a2a3] [--device 0]
          [--captured-at TS] [--rank LABEL] [--notes "…"] [--tags T …]
          [--copy] [--no-prune]
          [--bench "min=… median=… mean=… max=… rounds=…"]
          [--bench-log FILE]… [--modality-request NAME=VALUE]…
```

| 选项 | 说明 |
|---|---|
| `--program` | **程序名覆盖**。默认取自 name_map 文件名（通常是 `_jit_<case>_<hash>` 构建目录名）。要把同一段代码的多次采集当作同一程序做 compare/baseline，**必须**在每次 ingest 时统一传同一个 `--program`（见 §8.3 与 §12） |
| `--platform` / `--device` / `--captured-at` | 环境元数据 |
| `--rank` | 多卡采集时每个 rank 各 ingest 一次并打标签（如 `rank0`）；打了标签与没打标签的 run 不会混查 |
| `--notes` / `--tags` | 自由文本备注 / 标签数组 |
| `--copy` | 把工件复制进 `.pfdb/store/<run_id>/`（默认 link：只记路径 + sha256，不复制） |
| `--no-prune` | 跳过 ingest 后的自动 prune（**长战役必加**，见 §8.1） |
| `--bench` | 直接给一份 bench 汇总字符串（与 `--bench-log` 互斥） |
| `--bench-log` | 一个独立的 PYPTO_BENCH 日志 = 一个 bootstrap 分层（stratum），可重复传多次；接受对比需要 ≥3 层且两个 run 都有 |
| `--modality-request` | 覆盖 manifest 的模态请求状态（`NAME=true/false/值`），仅在 manifest 缺失或不准时应干预 |

完整实例（真实输出）：

```bash
export PFDB_PATH=$PWD/.pfdb/demo/profile.duckdb
pfdb ingest build_output/_jit_decode_csa_tp1_test_vh3vah7d/dfx_outputs \
    --program decode_csa_tp1 --platform a2a3 --device 0 --no-prune \
    --notes "decode_csa run A: level-4 swimlane" --tags demo
```

```text
ingested run 1 (program=decode_csa_tp1 level=4 mode=link)
  tasks=72 task_rows=4271 edges=242 artifacts=5 perf_hints=219 memory=0 pmu=0 makespan=3984.040us
  ingest_wall_ms=13386.781 ingest_cpu_ms=24472.001 peak_rss_bytes=1092128768 rss_source=getrusage.ru_maxrss_kib db_growth_bytes=0
```

逐字段解读：第一行是 run 身份（`level` 是采集级别，`mode` 是 link/copy）；第二行是各表行数与顶线指标（`makespan` 单位 µs；`perf_hints`/`pmu`/`memory` 是各模态入库条数，0 表示该模态无数据）；第三行是 ingest 自身遥测（耗时、CPU、峰值内存、数据库增量字节）。

要点：

- **幂等**：run 身份 = records 文件 sha256。重复 ingest 同一采集会原位替换该 run 的全部行（行数不变），不产生新 run；
- **同一事务**：解析、join、衍生层（密度带/空闲段/关键路径）一起提交，失败即整体回滚；
- **自动 prune**：ingest 成功后默认执行 `prune --keep 3`（可用 `--no-prune` 关闭）；
- level ≥ 2 的解析依赖 PyPTO 环境（复用上游 `read_perf_data` 做时钟域合并），请在 conda `pypto` 环境内操作；上游转换器拒绝的采集会以 `upstream converter rejected the capture: …` 报错。

### 4.2 `pfdb ingest-incore`

把 in-core 仿真器采集（目录内含 `manifest_export.csv`，可选 `instr_metrics.json`）挂到**已存在**的 run 上：

```bash
pfdb ingest-incore build_output/<case>/kernel_insight_all_funcs_<ts> --run 3
# in-core attached to run 3: 1 entries (0 exported)
```

- 只保存清单状态与指标汇总，原始大文件（`trace.clean.json`、`visualize_data.bin`、逐核 CSV）永不复制；
- 幂等：重复挂载先删旧行再插入；
- 对同一 run 重新 `ingest` 泳道会清掉 in-core 行，需要重新挂载；
- 查询：`pfdb query incore --run-id 3`。每个 kernel 一行，`status` 为 `exported` 时带各管线周期指标；失败时保留 `message`（示例见 §6.6）。

---

## 5. 读懂输出：facts DSL 与证据状态

### 5.1 三种输出格式

所有查询/list/compare/render 都支持 `--format facts|json|markdown`（默认 `facts`）：

```bash
$ pfdb query why_late --run-id 1 --task-id 12884901889
STALL dispatch_us=94.82 dispatch_wait_us=0.54 fin_detect_us=5.84 gap_us=8.12 ready_us=88.98 receive_us=95.36 run_id=1 start_us=97.1 start_wait_us=1.74 task_id="12884901889" upstream_depth=1 evidence=measured

$ pfdb query why_late --run-id 1 --task-id 12884901889 --format json
[{"rec":"STALL","dispatch_us":94.82,…,"evidence":"measured"}]

$ pfdb query why_late --run-id 1 --task-id 12884901889 --format markdown
| record | fields | evidence |
|---|---|---|
| STALL | dispatch_us=94.82 … | measured |
```

`facts` 格式：每行一条事实，`REC k=v …`，键按字典序、值 JSON 编码，行尾恒为 `evidence=<状态>`。

### 5.2 四种证据状态

| 状态 | 含义 |
|---|---|
| `measured` | 直接读自工件的时间戳/计数器 |
| `proven` | 由记录的确定结构/规则推导成立（如稀疏带归因、早派发证明、compare 兼容性注记） |
| `unproven` | 展示了相关性但不足以支持因果结论（如早派发"未证实"） |
| `unavailable` | 工件或字段不存在（level 不够、模态未采集、链头任务无生产者……） |

查不到的东西**永远返回 `unavailable` 事实，而不是猜测或空输出**：

```bash
$ pfdb query overview --run-id 99
RUN run_id=99 evidence=unavailable
```

### 5.3 字节预算与 TRUNCATED

每条查询有默认字节预算（§6 速查表列出了每条的默认值），`--budget N` 可覆盖。超预算时输出以一行显式收尾：

```text
TRUNCATED first_dropped_index=51 remaining=250 limit=4096 hint="retry --budget 32768"
```

`hint=` 给出的建议值就是"能放下当前全部结果"的预算，照做即可。`--format json` 也受同一预算约束（尾部带 `TRUNCATED` 标记记录），不存在绕过预算的格式。

### 5.4 固定阈值（不是查询参数）

| 阈值 | 值 | 用在 |
|---|---|---|
| 密度带存储宽度 | max(5 µs, 跨度/10000) | time_band 表 |
| 稀疏判定 | busy_cores ≤ 引擎核数 × 25% | `sparse` / `sparse_regions` / `why_sparse` |
| drain_tail | 最后一个 busy ≥ 50% 的带之后的带 | `drain_tail` 标记 |
| 空闲段记录门槛 | 同核相邻行间隔 ≥ 5 µs | idle_gap 表 / `core` 查询 |
| 早派发容差 | 2 个时钟周期（严格） | `early_dispatch` 证明 |
| scheduler 查询窗口 | 任务窗口 ± 1 µs | `scheduler` 查询 |

---

## 6. 查询体系：26 条查询逐条详解

所有 run 级查询都接受 `--run-id <id>`（必需）与可选 `--rank <label>`（与该 run 实际 rank 不一致会报错）。含下划线的查询名有连字符别名（`critical_path` ≡ `critical-path`）。`pfdb list` 是 `runs_list` 查询的薄封装。

**导航方法论**：不必按固定顺序走，跟着问题走——

- "现在库里有什么/哪个是最新可用 run" → Z0
- "时间都花在哪了/哪里空" → Z1
- "这个窗口/这颗核在干什么/这条带为什么空" → Z2
- "这个算子是谁、依赖谁、邻居多大" → Z3
- "它为什么启动晚/跑得久/哪条管线饱和" → Z4
- 模态查询按需取用。

以下示例除注明外来自手册示例库：run 1/2 = `hello_world`（level 4，16 任务），run 3 = 归档采集 `_jit_attention_csa_test`（level 4，**旧调度 schema**，71 任务，带 pmu/scope_stats/perf_hints/incore）——正好同时演示两代 schema 的兼容。

### 6.1 Z0 —— 定向

#### `pfdb list`（≡ `runs_list`）

列出工作集里每个 run 一行。参数：`--rank`（可选过滤）。

```text
RUN bench_mean_us=3685.488 device_id=0 edges=242 level=4 makespan_us=3984.04 platform="a2a3" program="decode_csa_tp1" rank="single" retained=true rows=4271 run_id=1 tasks=72 evidence=measured
```

字段：规模（tasks/rows/edges）、顶线（makespan_us、bench_mean_us）、`retained`（是否在工作集内）、`trial_id`（绑定的活跃 trial，无则缺省）。

#### `overview`

参数：`--run-id`。回答"这个 run 的顶线指标、拓扑与图规模"。

```text
$ pfdb query overview --run-id 1
RUN clock_freq_hz=50000000 device_id=0 git_commit="326322e…" git_dirty=true level=4 num_cores=60 platform="a2a3" program="hello_world" rank="single" retained=true run_id=1 evidence=measured
METRIC artifacts=4 bench_max_us=99.7 bench_mean_us=75.426 bench_median_us=75.2 bench_min_us=73.8 bench_rounds=300 cpm_us=51.12 edges=15 idle_gaps=9 makespan_us=96.12 raw_span_us=93.34 run_id=1 task_rows=16 tasks=16 time_bands=19 evidence=measured
RESOURCE cores=20 engine="aic" run_id=1 evidence=measured
RESOURCE cores=40 engine="aiv" run_id=1 evidence=measured
```

解读要点：先看 `level=`——低于 2 时 makespan/critical_path 等会是 `unavailable`；`cpm_us` 与 `makespan_us` 的比值给出"关键路径占比"；`clock_freq_hz=50000000` 表示 1 µs = 50 拍。

#### `inventory`

参数：`--run-id`。回答"这个 run 持有哪些工件、如何存储、能否重建"，以及每个模态的请求/解析状态：

```text
$ pfdb query inventory --run-id 3
ARTIFACT kind="chip_swimlane_records" rel_path="dfx_outputs/chip_swimlane_records.json" run_id=3 sha256="be13…" size_bytes=682518 store_mode="link" evidence=measured
ARTIFACT kind="incore_manifest" rel_path="manifest_export.csv" run_id=3 sha256="addd…" size_bytes=1542 store_mode="link" evidence=measured
…
MODALITY entry_count=31338 modality="pmu" parser_state="parsed" rel_path="dfx_outputs/pmu.csv" run_id=3 size_bytes=228779 state="available" evidence=measured
MODALITY modality="args_dump" parser_state="absent" reason="request state is unknown and artifact is absent" run_id=3 state="unknown_request" evidence=unavailable
```

模态 `state` 的取值与含义：

| state | 含义 |
|---|---|
| `available` | 已采集且解析成功（`entry_count` 给出行数） |
| `not_requested` | manifest 表明没请求过该模态且文件不存在 |
| `not_emitted` | 请求了但运行时没写文件（查采集 flag 与 manifest，不是 pfdb 的 bug） |
| `unknown_request` | 无 manifest 可判定（老采集），文件也不存在 |
| `empty` / `parse_error` | 文件在但为空 / 解析失败 |

### 6.2 Z1 —— 宏观密度

#### `density`

参数：`--run-id`、可选 `--engine aic|aiv`、`--bands N`（显示分桶数，默认 20）。回答"时间去哪了"。

```text
$ pfdb query density --run-id 1 --engine aiv
BAND band_idx=0 busy_cores=1 drain_tail=false engine="aiv" run_id=1 sparse=true t0_us=82.22 t1_us=87.22 task_ids=["12884901888"] total_cores=40 evidence=measured
…
```

每带给出时间窗、忙核数/总核数、覆盖的任务、是否稀疏、是否 drain 尾。`--bands` 只影响显示分桶；若要把带号传给 `why_sparse --band`，两个命令必须用同一个 `--bands`。

#### `sparse_regions`

参数：`--run-id`、可选 `--engine`、`--top-k`（默认 5）。按忙核数升序给出最空的存储带（**5 µs 存储带**，不是 density 的显示分桶），并直接给出归因：

```text
$ pfdb query sparse_regions --run-id 1 --top-k 3
SPARSE busy_cores=0 engine="aiv" fin_us=88.98 kind="ready_starved" lagging_producer="12884901888" run_id=1 stored_band_idx=1 t0_us=87.22 t1_us=92.22 total_cores=40 evidence=proven
```

`kind` 三种：`dispatch_wait`（有任务就绪但没被派发，`ready_task_ids` 列出）/ `ready_starved`（上游没喂上，`lagging_producer` + `fin_us`）/ `unknown`。把 `stored_band_idx` 原样传给 `why_sparse --stored-band` 可复现同一归因。

### 6.3 Z2 —— 区域与核

#### `why_sparse`

参数：`--run-id`，**二选一** `--band N`（density 的显示分桶号，需配 `--bands` 与 density 一致）或 `--stored-band N`（sparse_regions 的存储带号），可选 `--engine`。不要混用两种坐标。

```text
$ pfdb query why_sparse --run-id 1 --stored-band 1 --engine aiv
SPARSE busy_cores=0 engine="aiv" fin_us=88.98 kind="ready_starved" lagging_producer="12884901888" run_id=1 stored_band=1 t0_us=87.22 t1_us=92.22 evidence=proven
```

#### `region`

参数：`--t0-us`、`--t1-us`（必需），可选 `--family`、`--core`。回答"这个时间窗内发生了什么、为什么有空"：

```text
$ pfdb query region --run-id 1 --t0-us 82 --t1-us 97
REGION gaps=1 run_id=1 t0_us=82.0 t1_us=97.0 tasks=1 evidence=measured
TASK … task_id="12884901888" …
GAP core_index=22 engine="aiv" fin_us=88.98 kind="ready_starved" lagging_producer="12884901888" run_id=1 t0_us=86.14 t1_us=103.88 evidence=proven
```

#### `core`

参数：`--core N`（物理核号，必需）。回答"这颗核上跑了什么、哪里空"：

```text
$ pfdb query core --run-id 1 --core 20
CORE core_index=20 engine="aiv" gaps=4 rows=5 run_id=1 evidence=measured
ROW core_index=20 dispatch_us=94.82 end_us=101.18 finish_us=101.9 receive_us=95.36 row_index=1 run_id=1 start_us=97.1 task_id="12884901889" evidence=measured
…
GAP core_index=20 engine="aiv" fin_us=107.38 kind="ready_starved" lagging_producer="12884901890" run_id=1 t0_us=101.18 t1_us=114.58 evidence=proven
```

#### `idle_window`

参数：`--after-task-id`（必需）、可选 `--until-task-id`（默认：该生产者在观测关键路径上的下一个后继）、`--engine`（默认：生产者引擎的对面，aic↔aiv）。回答"生产者完成到消费者启动之间，对面引擎忙不忙"：

```text
$ pfdb query idle_window --run-id 1 --after-task-id 12884901888
WINDOW after_task_id="12884901888" engine="aic" run_id=1 t0_us=86.14 t1_us=97.1 until_task_id="12884901889" window_us=10.96 evidence=measured
OCCUPANCY busy_core_us=0.0 capacity_us=219.2 cores=20 engine="aic" occupancy=0.0 run_id=1 evidence=measured
```

解读：`occupancy` 是**相关性**证据，不是资源阻塞的证明——本例 aic 全程空闲（0.0），说明"把后续工作挪进这个窗口"不会撞上对面引擎。

### 6.4 Z3 —— 单算子与依赖

#### `task`

参数：`--task-id`（必需）。一个任务的身份、时序与关键路径归属：

```text
$ pfdb query task --run-id 1 --task-id 12884901889
TASK block_num=1 busy_us=4.08 early_dispatch_flag=false engine="aiv" family="add_scalar" kernel_ids=[-1,0,-1] max_end_us=101.18 max_finish_us=101.9 min_dispatch_us=94.82 min_receive_us=95.36 min_start_us=97.1 name="add_scalar" num_rows=1 on_cpm_observed=true on_cpm_static=true run_id=1 scope="auto" task_id="12884901889" wall_us=7.08 evidence=measured
```

口径：`busy_us = max(end) − min(start)`，`wall_us = max(finish) − min(dispatch)`；`on_cpm_observed/static` 标记是否在两条关键路径上。

#### `tasks`

参数：`--family`（精确名）或 `--name`（子串）**至少一个**（拒绝全图倾倒），可选 `--engine`、`--on-cpm observed|static`。按 task_id 数值排序返回一族任务，用于给 `task`/`pmu`/`why_long` 选代表：

```text
$ pfdb query tasks --run-id 3 --family qk_pv
TASK block_num=24 busy_us=589.7 early_dispatch_flag=true engine="mixed" family="qk_pv" kernel_ids=[42,43,43] … name="qk_pv_aic+qk_pv_aiv" num_rows=72 on_cpm_observed=true … task_id="4294967377" wall_us=590.66 evidence=measured
```

预算超限会显式 TRUNCATED（默认预算 16384）。

#### `deps`

参数：`--task-id`、`--direction in|out|all`（默认 `out`）。直接依赖边及其携带的张量元数据：

```text
$ pfdb query deps --run-id 1 --task-id 12884901889 --direction all
DEP arg="1" consumer_dtype="FLOAT32" consumer_shape=[1024,512] consumer_start_offset="0" consumer_strides=[512,1] flags=["wait"] pred="12884901888" run_id=1 source="tensormap" succ="12884901889" tensor_id="15348436186529585625" evidence=measured
DEP arg="1" … pred="12884901889" … succ="12884901890" … evidence=measured
```

`source` 取值：`explicit`（显式依赖）/ `creator`（host 创建者）/ `tensormap`（张量映射冲突）/ `host` 等；`flags` 常见 `wait`、`retain`。查生产者用 `--direction in`。

#### `subgraph`

参数：`--task-id`、`--depth`（1–6，默认 2）、`--max-nodes`（1–200，默认 24）。BFS 邻域：

```text
$ pfdb query subgraph --run-id 1 --task-id 12884901889 --depth 1
SUBGRAPH capped=false depth=1 nodes=3 run_id=1 task_id="12884901889" evidence=measured
NODE depth=-1 engine="aiv" family="add_scalar" name="add_scalar" run_id=1 task_id="12884901888" evidence=measured
NODE depth=0 … task_id="12884901889" …
NODE depth=1 … task_id="12884901890" …
DEP … pred="12884901888" succ="12884901889" …
```

host 侧创建者伪节点以 `kind=external` 出现；直接生产者的 depth 为 −1。

### 6.5 Z4 —— 微归因（需要 level ≥ 2）

level-1 采集上这些查询返回 `unavailable`：含义是"采集里没有 FIN/派发流"，**不是**"这个任务没有等待"。

#### `why_late`

FIN → dispatch → receive → start 四段分解：

```text
$ pfdb query why_late --run-id 1 --task-id 12884901889
STALL dispatch_us=94.82 dispatch_wait_us=0.54 fin_detect_us=5.84 gap_us=8.12 ready_us=88.98 receive_us=95.36 run_id=1 start_us=97.1 start_wait_us=1.74 task_id="12884901889" upstream_depth=1 evidence=measured
```

恒等式：`gap_us = fin_detect_us + dispatch_wait_us + start_wait_us`（本例 5.84+0.54+1.74=8.12）。`ready_us` = 直接生产者最晚 FIN；`upstream_depth` = 最长生产者链深。链头任务（无生产者）返回 `unavailable, upstream_depth=0`。

#### `why_long`

与同族任务对比：

```text
$ pfdb query why_long --run-id 1 --task-id 12884901889
LONG busy_us=4.08 family="add_scalar" family_median_us=3.0 family_rank=15 family_tasks=16 max_row_us=4.08 min_row_us=4.08 name="add_scalar" num_rows=1 run_id=1 task_id="12884901889" wall_us=7.08 evidence=measured
```

大任务示例（decode_csa 的 72 行混合核任务）：`busy_us=798.38, family_rank=1, family_tasks=1, min_row_us=353.38, max_row_us=444.08`——busy 是跨核聚合，行级 min/max 给出核间散差。

#### `rows`

任务的逐物理行时序（每行一个 ROW 事实：core_index、row_index、start/end/dispatch/receive/finish，µs）。

#### `scheduler`

任务窗口（±1 µs）内重叠的调度相与编排相：

```text
$ pfdb query scheduler --run-id 1 --task-id 12884901889
SCHED kind="dispatch" lane=0 loop_iter=35 run_id=1 t0_us=93.34 t1_us=95.44 tasks_processed=1 evidence=measured
SCHED kind="complete" lane=0 loop_iter=43 run_id=1 t0_us=101.66 t1_us=102.76 tasks_processed=1 evidence=measured
…
ORCH lane=0 run_id=1 submit_idx=1 task_id="12884901889" evidence=measured
```

`kind` 全集 14 种：`complete` / `dispatch` / `release` / `dummy` / `early_dispatch` / `resolve` / `resolve_standalone` / `dummy_task` / `predicated_skip` / `drain` / `drain_prepare` / `drain_publish` / `async_poll` / `graph_prepare`。大任务窗口会命中大量相（预算超限显式 TRUNCATED）。需要完整相数据请用 level 3/4 采集。

#### `early_dispatch`

早派发四态证明（`full` / `partial` / `none` / `unavailable`）：

```text
$ pfdb query early_dispatch --run-id 1 --task-id 12884901889
EARLY proven_blocks=0 run_id=1 status="none" task_id="12884901889" total_blocks=1 evidence=unproven
```

真实被证明的例子（decode_csa，24 块任务、7 块被时间戳证明提前派发）：

```text
EARLY proven_blocks=7 ready_us=849.04 run_id=1 status="partial" task_id="8589934625" tol_us=0.04 total_blocks=24 evidence=proven
```

`status=none` + `evidence=unproven` 表示"结构上符合条件但时间戳不支撑（或相反）"，与 `unavailable`（没有判定所需数据）区分。

#### `pmu`

参数：`--task-id`、`--samples`（附加逐样本事实，默认关）。按任务聚合各管线忙周期占比（默认预算 32768）：

```text
$ pfdb query pmu --run-id 3 --task-id 4294967377
MODALITY entry_count=31338 modality="pmu" parser_state="parsed" … state="available" evidence=measured
PMU_SUMMARY counters=9 measurements=648 run_id=3 samples=72 task_id="4294967377" evidence=measured
PMU counter="cube_busy_cycles" ratio=0.02004 run_id=3 samples=72 task_id="4294967377" total_cycles=40957277.0 value=820800.0 evidence=measured
PMU counter="mte2_busy_cycles" ratio=0.258989 … value=10607490.0 evidence=measured
PMU counter="scalar_busy_cycles" ratio=0.124844 … evidence=measured
PMU counter="pmu_total_cycles" ratio=1.0 … evidence=measured
```

`ratio = value / total_cycles`，瓶颈管线应接近 1.0。`pmu.csv` 无 `*total*cycle*` 列时 `ratio` 缺失并以 `EVIDENCE metric="ratio" reason="pmu.csv carries no total-cycles column" evidence=unavailable` 说明。

#### `critical_path`

参数：`--kind observed|static`（默认 observed；默认预算 32768）。observed = 实际执行的反向归因路径；static = 时长加权的最长依赖路径（无限核假设，交叉验证用）：

```text
$ pfdb query critical_path --run-id 1 --kind observed
PATH busy_us=3.92 compute_us=3.92 early_dispatch_proven="none" engine="aiv" family="add_scalar" gap_kind="front-gap" kind="observed" name="add_scalar" run_id=1 seq=0 stall_us=0.0 task_id="12884901888" wall_us=9.2 evidence=measured
PATH busy_us=4.08 … gap_kind="data-wait" gap_us=8.12 … seq=1 stall_us=10.96 task_id="12884901889" wall_us=7.08 …
…
```

每行：`wall/busy/compute/stall`（µs）、与前一任务的 `gap_us` 与 `gap_kind`（`data-wait`/`core-wait`/`front-gap`）、`early_dispatch_proven`。PATH 行自带 `name`/`family`，不需要再查一次 `task`。

#### `perf_hints`

编译器 tile/放置提示，逐字原样（compiler 起源的证据，报告时保持原文）：

```text
$ pfdb query perf_hints --run-id 3 --budget 1200
PERF_HINT origin="compiler" run_id=3 seq=1 source_path="models/…/decode_indexer.py:160:5" text="[perf_hint PH-MR-001] MemoryReuse: software pipelining requested depth 2 …" evidence=measured
TRUNCATED first_dropped_index=1 remaining=174 limit=1200 hint="retry --budget 32768"
```

#### `memory`

各缓冲空间用量对硬件上限（来自编译期 `report/memory_after_AllocateMemoryAddr.txt`；新编译流程默认不再写该文件，无此文件时返回 `MEMORY run_id=N evidence=unavailable`）。

### 6.6 模态查询

#### `incore`

参数：可选 `--kernel`。每个 in-core kernel 一行：

```text
$ pfdb query incore --run-id 3
INCORE export_dir="…/funcs/qk_pv/export" kernel="qk_pv" metrics={"artifact_count":"0",…,"message":"StepError('command failed rc=2: …')"} run_id=3 status="failed" evidence=measured
```

`status=exported` 时 `metrics` 含各管线周期汇总（来自 `instr_metrics.json`）；失败时保留原始 message，便于定位是哪一步失败。

#### `args_dump`

参数：可选 `--task-id`、`--stage`。内核边界捕获的张量/标量元数据：

```text
$ pfdb query args_dump --run-id 2
MODALITY entry_count=25552 modality="args_dump" parser_state="parsed" rel_path="dfx_outputs/args_dump/args_dump.json" request_value="2" requested=true run_id=2 size_bytes=7773618 state="available" evidence=measured
ARGS arg_index=4 bin_size=0 dtype="BFLOAT16" kind="tensor" role="input" run_id=2 seq=0 shape=[512,64] stage="before_dispatch" task_id="4294967303" task_id_raw="0x0000000100000007" task_id_u64="4294967303" evidence=measured
```

`stage` 取值 `before_dispatch` / `after_completion`；`task_id_raw`（十六进制原样）与 `task_id_u64`（规范十进制）同时保留以便对账 PMU/泳道的拼写。`args.bin` 的 payload 永不入库（`bin_size` 只记字节数）。

#### `scope_stats`

参数：可选 `--site`。逐作用域 begin/end 记录（首行 `seq=0` 是资源容量元数据）：

```text
$ pfdb query scope_stats --run-id 2
SCOPE payload={"fatal":false,"dropped":0,"total":154,"task_window_max":[16384,…],…} run_id=2 seq=0 evidence=measured
SCOPE payload={"depth":1,…} phase="begin" ring=1 run_id=2 seq=2 site="decode_csa_tp1_test.cpp:79" evidence=measured
…
```

#### `bench`

无观测基准数字（与 makespan 严格分口径）：

```text
$ pfdb query bench --run-id 2
BENCH max_us=3944.2 mean_us=3686.013 median_us=3663.05 min_us=3534.8 rounds=300 run_id=2 evidence=measured
BENCH_SAMPLE effective_us=3662.1 round=0 run_id=2 stratum=0 evidence=measured
…
```

`rounds` 是各分层轮数之和（3 × 100 = 300）；未注册 bench 时返回 `unavailable` 并提示 `ingest with --bench/--bench-log`。

---

## 7. 可视化渲染：pfdb render

```bash
pfdb render <kind> --run <id> [--t0 T --t1 T] [--task-id ID] [--core N] [--render-dir DIR]
```

| kind | 名称 | 参数 | 画什么 |
|---|---|---|---|
| `whole` | R0 全图 | 仅 `--run` | 全部物理核 × 全时间轴，一 task_row 一条横条，aic 蓝色 `#1f77b4`、aiv 橙色 `#ff7f0e` |
| `window` | R1 时间窗 | `--t0`、`--t1`（必需） | 窗口内行 + 依赖箭头（生产者结束→消费者启动，上限 200 条，超出在 manifest note 里说明）+ 虚线窗口边界 |
| `task` | R2 单任务邻域 | `--task-id` | 目标任务红色 `#d62728` 高亮，生产者/消费者淡化，绿色点线"就绪线" = max(生产者 FIN)（仅当存在真实 FIN 流才画） |
| `core` | R3 单核 | `--core` | 一颗核的时间轴，空闲段黄色 `#ffeb3b` 着色并标注 gap 类型（占轴 ≥5% 时） |

输出：800×450 PNG（matplotlib Agg 后端）+ 同名 `.manifest.json`（sha256、尺寸、µs/px、图例、生成器版本等）。CLI 结果是一个 `IMAGE` 事实：

```text
$ pfdb render whole --run 1 --render-dir .pfdb/demo-render
IMAGE cache_hit=false downsampled=false height=450 kind="whole" legend={"aic":"#1f77b4","aiv":"#ff7f0e"} path=".pfdb/demo-render/1/whole-6affb8b1b1c6be3b.png" run_id=1 sha256="803b…" size_bytes=68622 us_per_px=4.97605 wall_ms=7607.953 width=800 x0_us=75.72 x1_us=4056.56 evidence=measured
```

机制与注意事项：

- **缓存**：默认落在 `<数据库所在目录>/render/<run_id>/<kind>-<params_key>.png`，`params_key` 是 (kind, run_id, 渲染器版本, **run 数据指纹**, 参数) 规范化 JSON 的 SHA-256 前 16 位——run 指纹就是定义 run 身份的 records 文件 sha256，因此**同参数重复渲染逐字节一致并 `cache_hit=true`；同目录放多个数据库、或删库重建后 run_id 被复用，都不会命中另一个采集的旧图**（不同数据 ⇒ 不同指纹 ⇒ 不同键；0.3.0 及之前的旧缓存条目只会变成 miss，不会串图）。manifest 中记录 `run_fingerprint` 字段。总缓存 200 MiB LRU 逐出；损坏条目（sha 不符）直接弃用。
- **字节预算**：单图 1 MiB，超限自动降 DPI 重渲染并在 manifest 标 `downsampled`。
- **目标不存在**不报错：返回 `IMAGE … evidence=unavailable`（exit 0），`reason` 说明原因。
- 图像判断：先用文字类查询（critical_path + pmu 足以区分管线瓶颈/访存瓶颈）；当"某个命名窗口内的占用"仍不清晰时再渲染 R1/R2 看图。

---

## 8. 生命周期与短期记忆

### 8.1 `pfdb prune`：工作集清理

```bash
pfdb prune --keep 3      # 默认 keep=3
```

保留集 = 最新 `keep` 个 run + **每个 baseline 引用的 run** + **每个活跃（`status=running`）trial 绑定的 run**。其余删除（16 张 run 级表在同一事务内清空，随后删除 copy 归档与渲染缓存目录；link 模式不删任何源文件）。`trial`/`baseline` 行本身永不删除。

> **长战役必读**：ingest 后默认自动 prune（keep=3）。第 4 次采集入库时最老的 run 就会被删——除非它是 baseline 或活跃 trial。五个 trial 的循环如果第一次 ingest 就忘了 `--no-prune`，基线可能已经没了。

实测示例（3 个 run，run 1 受 baseline 保护）：

```text
$ pfdb prune --keep 1
pruned 1 run(s) [2]; kept [1, 3]
```

被删的 run 永远可以从 `build_output/` 重新 ingest 重建（会得到新的 run_id）。

### 8.2 `pfdb note`

```bash
pfdb note 2 "run B carries pmu/scope_stats/dump_args; timing perturbed by observers"
# note set on run 2
```

覆盖式设置 run 的自由文本备注。

### 8.3 `pfdb compare`：中性前后对比

```bash
pfdb compare <run_a> <run_b> [--family [NAME]] [--bootstrap --confidence 0.95 --resamples 10000 --seed 0]
```

**兼容门禁**：program、swimlane 级别、时钟频率、核数/核型全部相同才可比，否则拒绝：

```text
pfdb: error: runs 1 and 2 are not comparable: program differs: 'A' vs 'B'
```

⚠️ program 默认取自 `_jit_<case>_<hash>` 构建目录名，**同一段代码的两次采集 hash 不同**——要对比就在 ingest 时统一 `--program`（§4.1）。

输出（真实示例）：

```text
COMPARE compatible=true program="decode_csa_tp1" run_a=1 run_b=2 evidence=measured
NOTE bench_mean_us="unprofiled" makespan_us="profiled-with-observer" topic="metric-scope" evidence=proven
DELTA after=3686.013 before=3685.488 delta=0.525 metric="bench_mean_us" ratio=1.0 run_a=1 run_b=2 evidence=measured
DELTA after=766.68 before=798.38 delta=-31.7 family="qk_pv" metric="family_busy_us" ratio=0.96 run_a=1 run_b=2 evidence=measured
…
CONFIDENCE baseline_mean_us=3685.488 candidate_mean_us=3686.013 ci_high=0.003 ci_low=-0.003 confidence=0.95 metric="bench_mean_speedup" resamples=2000.0 samples_per_stratum=[100,100,100] seed=0.0 speedup=-0.0 strata=3.0 evidence=measured
```

- run 级 DELTA 恒有：`bench_mean_us`、`makespan_us`、`raw_span_us`、`cpm_us` + 三个规模计数；
- `--family`（裸 flag = 全部 family）追加每族 `busy/wall/tasks` DELTA，按 |busy Δ| 排序（预算默认提到 16384）；`--family qk_pv` 限定单族。整体 wall 差异不大时，family 层面可能藏着此消彼长——下结论前先看 family；
- `--bootstrap` 输出 `CONFIDENCE`：以独立调用为分层（stratum）的确定性自助法置信区间，对 `1 − candidate/baseline`。要求**两个 run 各有 ≥3 个 bench 分层**（`--bench-log` 传 3 次以上），`seed` 固定保证可复现；
- `NOTE` 事实反复强调口径：bench 无观测、makespan 带观测——**不要把 makespan 下降读成加速**，裁决看 bench。

### 8.4 `pfdb baseline`

```bash
pfdb baseline add 1 --name csa-baseline --bench-mean 3686.0
# baseline 1 added (name=csa-baseline run=1)
pfdb baseline list
pfdb baseline diff 2 --baseline csa-baseline   # 省略 --baseline 用最新注册的
```

baseline 用 run 的 bench_mean（或显式 `--bench-mean`）登记命名基准，并**保护该 run 不被 prune**。`diff` 的输出与 compare 相同（带 `baseline=`/`baseline_run_id=` 标记），同样受兼容门禁与 bootstrap 规则约束。

### 8.5 `pfdb trial`：调优实验闭环

```bash
pfdb trial register --goal "reduce tail" --hypothesis "earlier dispatch helps" \
    --changed-files models/x/k1.py --parent 1        # 血缘可回溯
pfdb trial bind 1 7                                   # trial 1 ← run 7
pfdb trial verdict 1 --verdict win --evidence "bench CI [2.1%, 3.5%]"
pfdb trial list [--active]
```

- `verdict` 四选一：`win | neutral | regression | compile_error`；pending 状态不能直接改判（编译失败 → `compile_error`，无需 bind）；
- 绑定的 run 在 trial 活跃期间受 prune 保护，出结论后解除；
- 数据库**永不**自己判断胜负——verdict 是你做完决定后的记录动作；
- **没有 profile run 也能用**：编译都没过 → `register` + `verdict compile_error`；只跑了 bench 没采泳道 → `register` + `attach-bench` + `verdict`：

```bash
pfdb trial attach-bench 2 --bench-log a.log --bench-log b.log --bench-log c.log
```

`trial list` 里 `bench_mean_us` 优先取 trial 自附的 bench，其次取绑定 run 的。

---

## 9. Agent 通道：MCP 服务

```bash
pfdb serve --mcp                 # stdio 传输，会话级生命周期（客户端断开即退出）
pfdb serve --mcp --writable      # 允许 trial/baseline/note 写操作（持有写锁！）
pfdb serve --mcp --path /abs/path.duckdb
```

- 不带 `--mcp` 直接 `pfdb serve` 会拒绝（exit 2）；
- 默认只读；`--writable` 打开写连接并**占用唯一的写者席位**（期间其他 ingest 会 LockError）；
- 错误以 `pfdb: error: …` 文本返回，不抛栈。

### 9.1 工具清单（38 个）

- **26 个查询工具**：由查询注册表自动生成，命名 `pfdb.<query>`（`runs_list` 改名 `pfdb.list_runs`），`inputSchema` 与 CLI 参数同源（pydantic），工具描述即该查询的 owner question；
- **10 个生命周期工具**：`pfdb.compare`、`pfdb.baseline_diff/list/add`、`pfdb.register_trial`、`pfdb.bind_trial`、`pfdb.attach_bench`、`pfdb.set_verdict`、`pfdb.list_trials`、`pfdb.note`；
- `pfdb.render`：返回 IMAGE 事实 + `ImageContent`（PNG base64，多模态模型可直接读图）；
- `pfdb.version`：返回 tool schema 版本（当前 3）。

### 9.2 端到端验证客户端

仓库自带一个 mock agent，仅用 MCP 工具走完 Z0→Z4 全会话（list_runs → overview → density → why_sparse → region → task → deps → why_late）：

```bash
PFDB_PATH=$PWD/.pfdb/profile.duckdb python profile_db/examples/mock_agent.py
```

输出节选（38 个工具被发现，每步打印服务端返回的 facts 文本）：

```text
# 38 tools: pfdb.list_runs, pfdb.overview, pfdb.inventory, …, pfdb.version
== list_runs ==
RUN bench_mean_us=3685.488 … program="decode_csa_tp1" … run_id=1 tasks=72 evidence=measured
…
# session complete (run_id=1)
```

接入你自己的 agent：把 `pfdb serve --mcp` 作为子进程拉起（stdio），会话结束自动退出。规则：只有会话里真的连上了 `pfdb.*` 工具才走 MCP；否则用 CLI + `PFDB_PATH`。

---

## 10. Python API

CLI 与 Python API 走同一引擎，`facts` 输出逐字节一致：

```python
from profile_db.api import ProfileDB, format_result

db = ProfileDB()                    # 解析 PFDB_PATH 或 <cwd>/.pfdb/profile.duckdb
r = db.query("overview", run_id=1)  # -> Result(facts, images, truncated)
print(format_result(r, "facts"))    # facts / json / markdown

img = db.render("whole", 1, render_dir=".pfdb/manual-render")
print(img.images[0].path)           # PNG 路径（供多模态模型读取）
db.close()                          # 释放写连接（如持有）

mem = ProfileDB.memory()            # 纯内存工作集（测试/临时分析）
```

`Result.images[i]` 提供 path、sha256、尺寸、图例映射。`ProfileDB` 也是制造 LockError 演示的方式：持有一个可写连接的同时从另一个进程 ingest，后者会立刻收到锁错误（见 §12）。

---

## 11. 数据库内部参考

### 11.1 表清单（schema v7）

| 表 | 内容 |
|---|---|
| `run` | 一次采集的上下文与顶线指标（platform/device/level/clock/拓扑/git/bench 汇总/makespan/cpm/notes/tags/retained/rank_label） |
| `artifact` | 工件清单（kind/rel_path/sha256/size/store_mode） |
| `task` | 每任务聚合（name/family/engine/scope/early_dispatch_flag/block_num/行数/busy/wall/min-dispatch…/CPM 归属；task_id 原样与 U64 规范双列） |
| `task_row` | 物理行时序（start/end/dispatch/receive/finish，µs） |
| `dep_edge` | 依赖边 + 张量元数据（source/arg/flags/dtype/shape/offset/strides） |
| `scheduler_phase` | 调度相（lane/kind/窗口/loop_iter/tasks_processed/pop_hit/pop_miss 计数/shared 队列深；task_id 列记录 dummy_task/predicated_skip/graph_prepare 作用的任务） |
| `orch_phase` | 编排 submit 相（lane/submit_idx/task_id/窗口） |
| `time_band` / `idle_gap` / `cpm_path` | 衍生层：5µs 密度带 / 核级空闲段 / observed+static 关键路径（ingest 时同一事务落表） |
| `pmu_counter` | PMU 长表（每计数器一行 + 原始样本坐标列） |
| `perf_hint` / `memory_entry` | 编译器提示逐行 / 缓冲占用 |
| `bench_sample` / `bench_stratum` | 逐轮原始样本 / 每次独立调用的分层元数据 |
| `args_dump_entry` / `scope_stats_entry` | 模态元数据（payload 类大对象不入库） |
| `modality_status` | 每模态的请求/解析状态（inventory 的数据源） |
| `incore_entry` | in-core 清单状态与指标 |
| `trial` / `baseline` | 调优实验 / 命名基准 |
| `schema_version` | 迁移版本 |

### 11.2 迁移机制

`profile_db/src/profile_db/schema/migrations/NNNN_name.sql` 按版本号顺序在事务内执行，失败整体回滚并保留旧版本号。升级数据库只需重跑任何命令——待执行迁移自动应用。

### 11.3 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功（含"目标不存在返回 unavailable 事实"的渲染） |
| 1 | 结构化错误（`pfdb: error: …`）：Ingest/Query/Render/Lifecycle/Db/Lock 等皆属此类 |
| 2 | argparse 用法错误、`serve` 缺 `--mcp` |

---

## 12. 故障排查（全部实测）

1. **"database is locked by another DuckDB process"（LockError）**
   另一个写者持有锁（ingest/prune/`serve --writable`/未关闭的 Python 写连接）。等待或结束它；数据库单写者是设计行为，**不要**用"删了 .pfdb"来"修"。

2. **"cannot open pfdb at …"（DbError）**
   文件不存在或损坏，或 `PFDB_PATH` 指向的不是你 ingest 的那个库。`pfdb init` 创建；删掉 `.pfdb/` 从 `build_output/` 重新 ingest 永远合法。**先 `echo $PFDB_PATH` 再怀疑别的**。

3. **"simpler_setup is unavailable" / "upstream converter rejected the capture"**
   level ≥ 2 的解析复用 PyPTO 环境的转换器。在 conda `pypto` 环境内 ingest；转换器拒绝的采集按报错信息检查采集完整性。

4. **ingest 报 "capture directory does not exist" / 缺 `deps.json` / `'edges' must be a non-empty list`**
   ingest 要求 records + deps.json（≥1 条边）+ 恰一个 name_map。泳道两遍采集的第一遍（dep_gen）失败时 deps.json 缺失——查采集日志中的 `prepare_native_run failed`；单任务无边的采集目前无法入库。

5. **compare 报 "program differs"**
   两次采集的 `_jit_<hash>` 构建目录名天然不同。ingest 时统一传 `--program <名字>` 即可归一（§4.1、§8.3）；报错信息会直接提示这一点。

6. **"stratified bootstrap requires at least three raw benchmark strata"**
   bootstrap 要求两个 run 各挂 ≥3 份 `--bench-log`（且日志含 `PYPTO_BENCH_RAW=1` 的逐轮样本行）。只挂了一个 run 也会报此错并指出两边的层数。

7. **makespan 与 bench 差异巨大**
   口径不同：makespan 带观测开销（level-4 + pmu/dump_args 等会把计时扰动到数倍），bench 无观测。**比较用 bench（bootstrap CI），解释用泳道**。真实案例：同一段代码，干净采集 makespan 3.98 ms，全模态采集 makespan 72.8 ms，而两者 bench 都是 ~3.69 ms。

8. **render 命中另一个采集的旧图（0.3.0 及之前的老问题，0.3.1 已修复）**
   老版本的缓存键不含 run 数据指纹，同目录多库共享 `.pfdb/render/` 时可能命中别的采集的图。0.3.1 起缓存键纳入 records sha256 指纹，此问题不再发生；升级后旧缓存条目只会安全地 miss（重新渲染），无需手动清理。

9. **`pfdb query xxx` 提示 invalid choice**
   查询名检查拼写；含下划线的名字有连字符别名（`critical-path`）。`pfdb query` 不带参数会列出全部合法名。

10. **查询输出戛然而止**
    不是 bug：字节预算 TRUNCATED 收尾行有 `hint="retry --budget N"`，照做即可（§5.3）。

---

## 13. 附录：速查表

### 13.1 CLI 速查

```bash
pfdb init [--path P]                     # 建库/迁移（幂等）
pfdb ingest <dir> [选项]                  # 采集 → run（幂等，自动 prune）
pfdb ingest-incore <dir> --run ID        # 挂 in-core 清单
pfdb list [--rank L]                     # 列 run（≡ query runs_list）
pfdb query <name> [参数]                  # 26 条查询
pfdb render <whole|window|task|core> --run ID […]
pfdb serve --mcp [--writable] [--path P] # MCP stdio 服务
pfdb prune [--keep N]                    # 工作集清理（默认 keep=3）
pfdb note <run> "<text>"                 # run 备注
pfdb compare <a> <b> [--family] [--bootstrap]
pfdb baseline add <run> --name N [--bench-mean F] | list | diff <run> [--baseline N]
pfdb trial register | bind | attach-bench | verdict | list
```

全局：`--version`；输出类：`--format facts|json|markdown`、`--budget N`；bootstrap 类：`--bootstrap --confidence F --resamples N --seed N`。

### 13.2 查询速查（默认字节预算）

| 层 | 查询（默认预算） | 一句话 |
|---|---|---|
| Z0 | `runs_list`(4096) | 工作集里有哪些 run |
| Z0 | `overview`(4096) | 顶线指标/拓扑/规模，先看 level |
| Z0 | `inventory`(4096) | 工件与模态状态 |
| Z1 | `density`(4096) | 分带占用 |
| Z1 | `sparse_regions`(4096) | 最空的带 + 归因 |
| Z2 | `why_sparse`(4096) | 一条带为什么空 |
| Z2 | `region`(4096) | 任意时间窗内容 |
| Z2 | `core`(4096) | 单核行与空闲段 |
| Z2 | `idle_window`(16384) | 生产者→消费者之间对面引擎忙不忙 |
| Z3 | `task`(4096) | 单任务身份/时序/CPM |
| Z3 | `tasks`(16384) | 按 family/name 选任务 |
| Z3 | `deps`(4096) | 直接依赖 + 张量元数据 |
| Z3 | `subgraph`(4096) | BFS 邻域 |
| Z4 | `why_late`(4096) | 四段启动延迟分解 |
| Z4 | `why_long`(4096) | 与同族比忙时 |
| Z4 | `rows`(4096) | 逐物理行时序 |
| Z4 | `scheduler`(4096) | 任务周边调度/编排相 |
| Z4 | `early_dispatch`(4096) | 早派发四态证明 |
| Z4 | `pmu`(32768) | 管线忙占比 |
| Z4 | `critical_path`(32768) | observed/static 关键路径 |
| Z4 | `perf_hints`(4096) | 编译器提示原文 |
| Z4 | `memory`(4096) | 缓冲占用对上限 |
| 模态 | `incore`(4096) | in-core 清单状态/指标 |
| 模态 | `args_dump`(4096) | 内核边界张量元数据 |
| 模态 | `scope_stats`(4096) | 作用域统计 |
| 模态 | `bench`(4096) | 无观测基准数字 |

### 13.3 相关文档

- 顶层设计：[profile_db/DESIGN.md](../DESIGN.md)
- 项目简介与状态：[profile_db/README.md](../README.md)
- 英文命令参考：`docs/debug-and-tune/profile-db.md`
- 采集命令与证据对照：`docs/debug-and-tune/profiling-options.md`
- agent 使用说明（skill）：`.agents/skills/profile-feedback/SKILL.md`
- 端到端实机演示：[demo-walkthrough.zh.md](./demo-walkthrough.zh.md)
