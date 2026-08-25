# 面向 Agent 的 Profiling 与 Feedback 数据库系统 · 顶层设计

> 状态：顶层设计稿（v0.3），待评审。
> 本文是后续逐步实现的总蓝图：先定架构与数据模型，再按第 11 节的任务分解逐个落地。
> 每个子任务都附带可自动验证的验收标准；文档语言为中文（经用户明确要求）。
>
> **v0.2 修订要点**（依据用户决策）：
> 1. 系统定位从"数据仓库"改为"**查询优先的工作台**"——profiling 数据强时效性，
>    每轮算子修改后旧数据大多失去价值、仅余对比之用；查询能力 > 存储能力，
>    历史数据主动清理而非持久归档；
> 2. `.pfdb/` 每仓库一份，纳入 git 忽略；
> 3. MCP 服务不常驻，kernel agent 工作时按需启动，仅需稳定接口；
> 4. schema 口径、band 粒度、稀疏阈值等细则由设计方决定，并用
>    `tools/calibrate.py`（原 scratch 探针收编而来）对真实 Qwen3Decode
>    捕获做了标定（标定结论见 6.3 条，原始数据见附录 B）。
>
> **v0.3 修订要点**：
> 5. 查询接口定性为"**信封稳定、目录演进**"：输出机制（Result/DSL/证据/预算）
>    冻结，具体查询名录以 kernel agent 的信息需求为唯一生长依据，开发过程中
>    逐步补充（新增 6.6 节，第 9 节与 T4 验收随之调整）。

---

## 0. 术语

| 术语 | 含义 |
|------|------|
| **工件（artifact）** | 一次编译/采集产出的原始文件，如 `chip_swimlane_records.json`、`deps.json`、`pmu.csv`。 |
| **运行（run）** | 一次完整的 profiling/benchmark 采集及其程序、平台、配置、代码版本等上下文。库以 run 为基本组织单位。 |
| **事实（fact）** | 查询返回给 agent 的最小知识单元，一行一条，恒带证据状态。 |
| **证据状态** | `measured`（直接读数）、`proven`（由结构与规则确定性推出）、`unproven`（仅相关，不足断言）、`unavailable`（工件缺失）。 |
| **衍生（derived）** | 由原始数据确定性计算出的索引与结论表（密度、空闲段、关键路径等），入库时算好，查询时不再解析原始 JSON。 |
| **工作集（working set）** | 当前有查询价值的 run 集合：最新若干次 run + 被基线/进行中试验引用的 run。之外的 run 可被 prune 清理。 |
| **时效性（decay）** | 一轮算子修改后，旧 profiling 数据的绝对性能意义失效，仅保留与基线的相对比较价值。 |
| **缩放层级（zoom level）** | 分层查询的粒度台阶，模仿人类读泳道图"整体 → 稀疏处 → 原因 → 算子 → 依赖"的阅读顺序。 |
| **查询目录（catalog）** | 具体查询的集合；每个查询必须绑定它所回答的 agent 问题（owner question）。目录可演进；输出信封（DSL/证据/预算）稳定。 |

---

## 1. 背景与目标

### 1.1 背景

性能调优是迭代优化的过程，profiling 与 feedback 是迭代的基础。当前 profiling
产物（见 2.1 节清单）存在四个问题：

1. **形态繁杂、分散**：L2 泳道原始记录、依赖图、调度相位、PMU、编译器报告、
   端到端 benchmark、in-core 模拟器痕迹等散落在 `build_output/` 各处，
   每次查阅都要重新扫描、解析、拼接。
2. **不利于 LLM 读取**：产物多为面向人类工程师（或可视化工具）的重结构 JSON。
   以泳道记录为例，它是视觉模态信息的文本载体——人类靠"看"图发现
   硬件利用不足、依赖阻塞、优化空间；让 LLM 直接读 JSON 既费 token 又抓不住结构。
3. **没有查询体系**：跨文件的证据（密度、稀疏、依赖、等待分解）没有持久索引，
   无法"先宏观后微观"地导航式查询。
4. **强时效性被当作持久数据对待**：每轮修改算子后，旧 profiling 几乎全部失去
   意义（仅剩与基线的对比价值），而现有产物只增不清理，查阅与存储成本随时间膨胀。

此前手写的 `profile_feedback.py` 是一个无状态的只读查询脚本，**仅作为概念参考**
（其"证据状态"与"字节预算"两个思路被本设计吸收），不迁移其代码，
也不需要与之兼容。本系统从头设计。

### 1.2 设计目标（按优先级）

1. **查询第一**：为 agent 提供从宏观到微观的分层查询体系（第 6 节），
   一句查询换一小段紧凑的带证据事实，这是本系统的主要价值。
2. **统一 ingest**：为工件提供归一化入库管道，一次 ingest、反复查询，
   并支持"删除后从 `build_output` 随时重建"。
3. **视觉模态专用处理**：泳道图作为视觉信息对象管理：语义结构化查询为默认通道，
   按需渲染图像为可选通道（供多模态模型）；**原始视觉 JSON 永不直接进入 LLM 上下文**。
4. **短期优化循环记忆**：记录当轮实验血缘（trial）、假设、基线与对比，
   支撑跨 run 对照；不做长期知识库。
5. **时效性管理与体积控制**：默认不复制原始文件；提供 prune 策略主动清理
   过期 run，防止文件体积膨胀。
6. **确定性、可验证**：所有衍生结论可复算；因果性断言必须携带证据状态，
   数据库只提供证据，不下诊断结论（调优决策仍由 agent/caller 做出）。

### 1.3 非目标（明确的"不做"）

- **不做采集**：数据库只 ingest 已有工件，绝不发起编译、设备运行或 profiling。
- **不做自动优化建议**：可以回答"哪里稀疏""为什么晚启动"，不给"应该怎么做"。
- **不做持久归档**：不承诺冷备份、迁移、长期历史查询；库是**可弃的工作集**，
  源工件在 `build_output`，库可由其随时重建。数据价值随时间衰减是设计前提。
- **不替代 Perfetto / pypto-toolkit** 等人类可视化工具；渲染层仅服务 agent
  与快速抽查。
- **不追求通用大数据平台**：面向单机、单项目、小团队规模（见 5.1 选型理由）。

---

## 2. 现状与差距

### 2.1 工件清单（数据源盘点）

| 类别 | 工件 | 模态 | 形态 | 实测规模（Qwen3Decode 捕获） |
|------|------|------|------|------|
| 泳道主记录 | `chip_swimlane_records.json`（旧名 `l2_swimlane_records.json` / `l2_perf_records.json`） | 视觉（承载） | JSON：元数据 + 每核每任务 cycle 时间戳 + AICPU 调度/编排相位 | 2549 行，706 物理行 |
| 依赖图 | `deps.json` | 结构/文本 | JSON：266 逻辑任务、2546 边 | 单行大 JSON |
| 名称映射 | `name_map_<Program>_<ts>.json` | 文本 | callable_id → kernel 名 | 小 |
| 合并轨迹 | `merged_swimlane_<ts>.json` | 视觉 | Perfetto traceEvents | 大，仅可视化用 |
| 编译器提示 | `report/perf_hints.log` | 文本 | 逐行 hint + 源码位置 | 小 |
| 内存占用 | `report/memory_after_AllocateMemoryAddr.txt` | 文本 | Vec/Mat/Acc 等空间 vs 硬件上限 | 小 |
| PMU | `pmu.csv` | 表格 | 每核每任务各管道 busy cycles（列名随架构可变） | 中等 |
| 端到端基准 | `PYPTO_BENCH` 输出（min/median/mean/max） | 数字 | stdout 或日志 | 微型 |
| in-core 模拟器 | `manifest_export.csv` + 各核目录 + `trace.clean.json` + `instr_metrics.json` | 视觉/表格 | per-instruction 管道痕迹 | 可能很大 |
| 数值定位 | `args_dump/`（args_dump.json + args.bin） | 二进制/表格 | 核边界张量捕获 | 可能很大 |
| 关键路径 | `CPM_observed.json` / `CPM_static.json` / `critical_path_report.md` | 结构/文本 | 关键路径任务与间隔分解 | 小 |

> 采集方式与各工件语义见 `docs/debug-and-tune/profiling-options.md`，本设计
> 只规定入库后的组织与查询。

### 2.2 现状差距（本系统要弥补的）

| 差距 | 现状 | 本系统 |
|------|------|--------|
| 组织 | 文件散落 `build_output/`，靠脚本每次扫描 | 归一化入库 + 索引，一次 ingest 反复查询 |
| 导航 | 一次查询一个切面，无导航结构 | 五级缩放（Z0–Z4）+ 专用 why 查询 |
| 跨运行 | 仅 `compare` 临时两两比较 | 短期血缘/基线/对比（不承诺长期历史） |
| 视觉模态 | 让 LLM 读 JSON 或拒绝处理 | 语义分层 + 按需渲染图像，双通道 |
| 体积 | 只增不减，无生命周期 | 默认不复制工件 + prune 主动清理 |
| 因果边界 | 证据状态只在输出格式里 | 进 schema、进类型、进测试，强制执行 |

---

## 3. 设计原则

1. **查询优于存储**：资源优先投在查询引擎与衍生层；存储做薄（link 为主、
   按需 copy、可整体丢弃重建）。查询性能与表达力是首要质量属性。
2. **LLM 优先**：一切输出默认是给 LLM 的：紧凑、结构化、带解释、受预算约束；
   人类可读版只是同一事实流的排版。
3. **证据第一**：所有事实挂证据状态；时间戳重合、占用率高低等**相关不等于
   因果**，衍生层不允许生成无证据的因果断言。
4. **时效性内置**：库承认数据衰减：工作集之外的 run 允许被 prune；
   与 baseline/trial 相关的对比价值受显式保护（retained 标记）。
5. **可重建**：库可从 `build_output` 工件随时重建；每次 ingest 记录工件
   SHA-256 与相对路径，重建结果与初次一致（确定性）。
6. **入库即衍生**：所有昂贵的解析/计算（密度、空闲段、关键路径、stall 分解）
   在 ingest 时完成并落表；查询只走视图与衍生表，**查询路径零 JSON 解析**。
7. **分层查询、逐层有界**：每层查询有默认返回量与字节预算，超限截断并显式
   标记 `TRUNCATED`，绝不静默丢数据。
8. **确定性**：同一库同一次 ingest，任何查询/渲染结果逐字节一致（可测试）。
9. **单向依赖**：相信"分层可维护性"（10.2 节依赖规则，CI 拦截违规）。
10. **可验证优先**：每个子任务以"可离线自动验证的验收标准"定义完成
    （无设备、无网络也可回归）。
11. **中立仓库**：库保存证据与短期记忆，不内置调优策略；策略属于 agent。
12. **渐进落地**：每个里程碑结束都留下一个可用、可演示的增量。

---

## 4. 总体架构

```text
                         编译/采集产物 (build_output/)
                                     │  ingest（适配器注册表，按工件类型挂插件）
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  PFDB 核心                                                         │
│                                                                   │
│  ① 工件引用层                     （默认 link 模式：记录相对路径 +    │
│                                    sha256，不复制文件；可选 copy）    │
│  ② 关系层 DuckDB                  （run/artifact/task/task_row/      │
│                                    dep/scheduler/.../trial/baseline）│
│  ③ 衍生层                         （密度带、空闲段、CPM、stall 分解、  │
│                                     early-dispatch 证明，全带证据标注）│
│  ④ 查询引擎（重心）                （Z0–Z4 分层 + why 查询 + 过滤 + 预算）│
│  ⑤ 视觉渲染层                      （4 级渲染 + 参数化缓存 + 清单）    │
│  ⑥ 生命周期层                      （工作集判定、prune、重建）         │
│  ⑦ 短期记忆层                      （trial / 假设 / 基线 / 对比）     │
└──────────────────────────────────────────────────────────────────┘
        │                                        │
        ▼                                        ▼
   Python API (公开)                     MCP tools (LLM function-calling)
        │                                输出: Result{ facts[], images[], truncated }
        ▼
   CLI `pfdb`（人 / 脚本）
```

一句话概括：**ingest 把工件变成库（薄），查询把库变成带证据的紧凑事实（重），
渲染把视觉区间变成小图，生命周期层随时可以把过期部分删掉重建。**

---

## 5. 数据模型（DuckDB，schema v1）

### 5.1 引擎、位置与生命周期

- **引擎**：DuckDB，嵌入式。默认库文件 `<项目根>/.pfdb/profile.duckdb`，
  可用环境变量 `PFDB_PATH` 覆盖；**每仓库一份**，`.gitignore` 增加 `.pfdb/`
  （T0 落地）。
- **查询优先的两档形态**：
  - 磁盘库：正常 ingest/查询/prune；
  - 内存库：`ProfileDB.memory()`，把一次 ingest + 一连串查询全部跑在
    `:memory:` 表上（同一套 schema/衍生/查询引擎），用完即弃，零磁盘痕迹。
- **存储做薄**：`artifact.store_mode` 默认 `link`（仅记录相对路径 + sha256，
  文件留在 `build_output`），显式 `copy` 才把文件拷入 `.pfdb/store/`。
  in-core 等大类工件永远只入指标/元数据，原始文件不复制。
- **清理**：`pfdb prune`（见第 8 节）删除超过工作集的 run 及其行；
  link 模式无文件可删，copy 模式连带删除副本。整个 `.pfdb/` 可随时删除，
  从 `build_output` 重建。
- **并发**：单写多读。写集中在 ingest（CLI 或 MCP 进程内的唯一写者）；
  查询一律只读连接，互斥由库级锁保证。

### 5.2 表清单

| 表 | 说明 | 关键列 |
|----|------|--------|
| `run` | 一次采集的上下文与顶线指标 | `run_id`、`program`、`platform`、`device_id`、`captured_at`、`swimlane_level`、`clock_freq_hz`、`num_cores`、`core_types/json`、`core_to_thread/json`、`rank_label`、`git_commit`、`git_dirty`、`runtime_cfg/json`、`cmdline/json`（脱敏）、`bench_min/median/mean/max_us`、`bench_rounds`、`makespan_us`、`raw_span_us`、`cpm_us`、`retained`、`notes`、`tags[]` |
| `artifact` | 工件清单与存档凭据 | `artifact_id`、`run_id`、`kind`、`rel_path`、`sha256`、`size_bytes`、`store_mode(link/copy)` |
| `task` | 逻辑任务（deps + 时序拼接后） | `run_id`、`task_id`、`name`、`family`、`engine`、`scope`、`early_dispatch_flag`、`kernel_ids/json`、`block_num`、`num_rows`、`busy_us`、`wall_us`、`min_dispatch_us`…`max_finish_us`、`on_cpm_observed`、`on_cpm_static` |
| `task_row` | 物理执行行（泳道每一格） | `run_id`、`task_id`、`core_index`、`engine`、`thread`、`row_index`、`start_us`、`end_us`、`dispatch_us`、`receive_us`、`finish_us`、`aux`（disp/receive/finish 由迁移 0003 补齐，level-1 维持转换器合成的 0.0） |
| `dep_edge` | 任务依赖边（原始字段保真） | `run_id`、`pred`、`succ`、`source`、`arg`、`flags/json`、`tensor_id`、`consumer_dtype`、`consumer_shape/json`、`consumer_start_offset`、`consumer_strides/json` |
| `scheduler_phase` | AICPU 调度相位 | `run_id`、`lane`、`kind(dispatch/complete/resolve/release)`、`t0_us`、`t1_us`、`loop_iter`、`tasks_processed`、`pop_hit`、`pop_miss`、`shared_at_start`、`shared_at_end`（后两者为**每队列深度的 JSON 列表**，迁移 0002 修正了 v1 误标 INTEGER） |
| `orch_phase` | AICPU 编排提交 | `run_id`、`lane`、`submit_idx`、`task_id`、`t0_us`、`t1_us` |
| `time_band`（衍生） | 密度索引：run 时间轴按固定粒度切带 | `run_id`、`band_idx`、`t0_us`、`t1_us`、`engine`、`total_cores`、`busy_cores`、`task_ids/json`、`sparse`、`drain_tail` |
| `idle_gap`（衍生） | 核级空闲段及确定性分类 | `run_id`、`engine`、`core_index`、`t0_us`、`t1_us`、`kind(dispatch_wait/ready_starved/drain_tail/unknown)`、`ready_task_ids/json`、`evidence`。<br>判定优先级按 6.3 逐条（dispatch_wait → ready_starved → drain_tail → unknown），1/2 类仅 level≥2 可用（level-1 无 FIN 流，占位 0.0 绝不解释为时刻）；payload 语义分 kind：dispatch_wait=就绪未派任务 id 列表，ready_starved=`[{task_id, fin_us}]` 滞后生产者，其余为空；三实类 `evidence=proven`，unknown `evidence=unproven`。记录阈值：同核相邻行间隔 ≥5µs（含边界）。 |
| `cpm_path`（衍生） | 关键路径任务序列与间隔分解 | `run_id`、`kind(observed/static)`、`seq`、`task_id`、`wall_us`、`busy_us`、`compute_us`、`stall_us`、`gap_us`、`gap_kind`、`early_dispatch_proven(full/partial/none/unavailable)`。<br>算法与上游 `critical_path` 逐条件同构（对拍即真）：observed 行 `compute_us`=非重叠实际贡献、`stall_us`=距 frontier 的间隔、`gap_kind∈data-wait/core-wait/front-gap`、`gap_us=start-ready`（level≥2 且生产者带时刻时）；static 行为依赖受限最长路径，`compute_us=busy_us`，gap 列空。early-dispatch 用结构规则（直接生产者全部 creator 或 allow_early_resolve）＋两 tick 容差时间戳证明。 |
| `pmu_counter` | PMU 长表（列名随架构动态） | `run_id`、`task_id`、`counter`、`value`、`total_cycles` |
| `perf_hint` | 编译器提示逐行 | `run_id`、`seq`、`text`、`source_path`、`origin='compiler'` |
| `memory_entry` | 缓冲占用报告 | `run_id`、`kernel`、`space(Vec/Mat/Left/Right/Acc)`、`usage`、`limit` |
| `bench_sample` | 可选原始基准样本 | `run_id`、`round`、`effective_us` |
| `incore_entry` | in-core 指标（仅指标入表） | `run_id`、`kernel`、`status`、`export_dir`、`metrics/json` |
| `trial` | 一次调优实验（短期） | `trial_id`、`parent_trial_id`、`run_id`、`goal`、`hypothesis`、`changed_files/json`、`status(running/done/abandoned)`、`verdict(win/neutral/regression/pending)`、`evidence_refs/json`、`created_at`、`notes` |
| `baseline` | 命名基线（受 prune 保护） | `baseline_id`、`name`、`program`、`platform`、`run_id`、`bench_mean_us`、`criteria/json`、`accepted_at` |
| `schema_version` | 迁移版本记档 | `version`、`applied_at` |

索引：`task(family)`、`task(run_id, task_id)`、`task_row(run_id, core_index, start_us)`、
`dep_edge(pred)`、`dep_edge(succ)`、`time_band(run_id, engine, band_idx)`、
`trial(run_id)`、`cpm_path(run_id, kind, seq)`。

视图：`v_run_summary`、`v_family_stats(run)`、`v_region(t0,t1)` 等，只读、
只引用表格。

### 5.3 口径决策表（schema 细则，已定）

| 项 | 决策 | 依据/备注 |
|----|------|-----------|
| 拓扑与时钟 | 全部从 artifact `metadata` 读（`num_cores`/`core_types`/`core_to_thread`/`clock_freq_hz`），**绝不硬编码** | 标定发现不同捕获核心数不同（本次 20 AIC + 40 AIV，此前文档示例为 24+48） |
| 时钟域合并 | 只复用 `simpler_setup.tools.swimlane_converter.read_perf_data`；禁止自研双时钟域拼接 | 风险表第 12 节 |
| `makespan_us` | `max(finish) - min(dispatch)`（converter 拼接后口径） | 与既有分析器/关键路径工具同源；另一口径另存 `raw_span_us`（原始 aicore 行跨度，实测 2825.18µs） |
| 任务时长 | 双列：`busy_us = max(end) - min(start)`（行级）；`wall_us = max(finish) - min(dispatch)`（生命周期） | 消除"duration"歧义 |
| `ready(task)` | `max(FIN(直接生产者))`，边语义（`flags`、artifact 边）以 simpler 运行时口径为准；T1 与 `critical_path` 工具对拍锁死后固化测试常量 | 不臆造边语义 |
| 密度带粒度 | 存储粒度 **5µs**（`time_band`），自适应：`res = max(1µs, span/10000)`（防超长 run 表膨胀）；查询时按 `--bands N` 聚合展示 | 标定：AIV 任务中位 11.9µs、最小 1.8µs，5µs 以下粒度才能保留短任务信号（附录 B） |
| 稀疏判定 | 带内 `busy_cores ≤ 25% × 该引擎核数` 且非 `drain_tail` 后缀 → 稀疏候选；`drain_tail` = 最后一个"忙碌 ≥50% 核数"的带之后的后缀 | 标定：AIV 44% 的 5µs 带为空、62% ≤ 10 核；AIC 中位带宽即全忙（附录 B） |
| 空闲段记录阈值 | 同核相邻两行间隔 ≥ **5µs** 记入 `idle_gap`；≥ **1µs** 在输出中标记为"显著间隔"（沿用既有临界路径报告惯例） | 标定：AIC 间隔中位 0.94µs，AIV 中位 40.9µs |
| 基准 vs 观测开销 | `bench_*`（`PYPTO_BENCH`，无观察者开销，头指标取 mean）与 `makespan_us`（带观察者开销）严格分列，永不混用 | profiling-options.md 既有语义 |
| rank | `rank_label` 默认 `'single'`；多 rank 捕获必须按 rank 隔离行集 | 防止静默混 rank |
| 路径与脱敏 | 入库路径一律相对路径；任何环境元数据不含机器用户名 | 仓库既有政策 |
| 边数量 | `dep_edge` 逐条保真入库（实测 2546 条），字段名以真实工件为准：`arg/consumer_dtype/consumer_shape/consumer_start_offset/consumer_strides/flags/pred/source/succ/tensor_id` | 标定脚本读出 |

### 5.4 事实 DSL v2

查询输出的行式格式（机器导向；类型名保持 ASCII 惯例，值允许中文/Unicode）：

```text
RUN     run_id=1 program=Qwen3Decode platform=a2a3 ...
METRIC  run_id=1 makespan_us=1868.560 cpm_share=0.790 ...
BAND    run_id=1 band_idx=17 engine=aiv busy_cores=1 total_cores=40 ...
SPARSE  run_id=1 band_idx=17 engine=aiv kind=ready_starved evidence=proven ...
TASK    run_id=1 task_id=4294967298 name=q_proj family=q_proj engine=aic busy_us=...
PATH    run_id=1 kind=observed seq=4 task_id=... wall_us=... stall_us=...
GAP     run_id=1 core=27 t0_us=1200.4 t1_us=1311.0 kind=dispatch_wait ...
STALL   run_id=1 task_id=... fin_detect_us=... dispatch_wait_us=... start_wait_us=...
...
EVIDENCE artifact=deps.json status=unavailable
TRUNCATED limit_bytes=4096
```

结构值（shapes、计数器映射、编译文本）一律 JSON 编码，避免二次转义协议。
**任何 fact 必须且只能有一个证据状态**；无证据时输出 `EVIDENCE ... status=unavailable`，
绝不估值。

### 5.5 版本与迁移

- `schema_version` 全局记档 + `migrations/NNNN_*.sql` 顺序迁移，连接打开时自动执行。
  已有迁移：`0001_init.sql`（18 表全量建表）、`0002_sched_queue_depths.sql`
  （`scheduler_phase.shared_at_*` 改 JSON 列表——T1 对真实捕获的保真修正）、
  `0003_task_row_dispatch.sql`（`task_row` 补 `dispatch/receive/finish_us`
  三列——T3 行级 early-dispatch 证明与最早行 stall 分解所需）。
- schema 自 v1 起**一次性预留**全部表（含 trial/baseline 与衍生表），避免后期大迁移。
- 库文件可整体删除重建（数据可弃），因此不提供降级/回滚语义——迁移只前进。

### 5.6 技术栈选型（已定）

**主语言：Python（≥3.10），全链路唯一实现语言；分析引擎：嵌入式 DuckDB
（其内核本身是 C++ 向量化引擎）。**

理由：

1. **生态对齐**：MCP SDK、pydantic（tool schema 单一同源）、LLM agent 工具链
   都是 Python 原生。若用 C++ 做主体，接口层必然要 bindings 或常驻侧车
   进程，复杂度不降反升。
2. **正确性依赖在 Python 侧**：L2 时钟域合并必须复用
   `simpler_setup.tools.swimlane_converter.read_perf_data`（设计明文禁止
   重写该逻辑），CPM 等上游语义对齐也以 Python 工具为对拍基准——
   Python 主语言才能零成本复用。
3. **数据规模不构成 C++ 理由**：工作集量级是"266 任务 / 2546 边 / 706 行"，
   in-core 等大工件只存指标。查询瓶颈在表达力与 LLM 上下文经济，
   不在计算吞吐。
4. **重活已在库里**：所有关系型存储与分析落 DuckDB，Python 只是薄壳；
   衍生计算优先写成 DuckDB SQL（必要时 polars/pyarrow），把最热的路径
   推给 C++ 内核执行。
5. **可维护性**：单语言、单仓库、pytest 全绿即验收，与仓库现有工具同构，
   符合"可验证优先"原则。

边界与逃生通道：

- 衍生层默认 SQL/声明式 Python 实现并落表；
- 若未来实测某个衍生器成为性能热点（先 benchmark 再说话），可在
  **schema 与 API 不变**的前提下局部改 DuckDB UDF 或单个 C++ 扩展模块——这是
  逃生通道，不是默认路径；
- 渲染层用 matplotlib（确定性样式），无 C++ 诉求；
- 明确排除：第二语言作主实现、常驻 C++ 服务进程。

---

## 6. 分层查询体系（核心交付，系统重心）

### 6.1 阅读学

人类看泳道图的路径是：

1. 先看**整体**：哪些引擎、利用率多少、哪里密集哪里稀疏；
2. 再看**稀疏带**：为什么这段核都空着？没任务可发，还是调度没来得及发？
3. 点开一个**算子**：它的依赖是什么？上游谁最晚结束？
4. 追问**为什么不能提前**：最后一个上游 FIN 到自己的 START 之间发生了什么。

本系统把这条路径固化为五个缩放层级 + 一组 `why` 专查。每层返回
**小、有界、可导航**的结果（任何事实都携带下一步可查的 `run_id / task_id /
band / core` 坐标）。

### 6.2 缩放层级

| 层级 | 语义 | 典型查询 | 返回 |
|------|------|----------|------|
| **Z0 库与运行** | 有哪些 run、什么配置、哪里来的 | `runs list`、`overview <run>`、`inventory <run>` | run 元数据、工件清单、顶线指标（makespan/CPM/利用率/图规模） |
| **Z1 宏观密度** | 整图哪里密集哪里稀疏 | `density <run> [--engine aiv] [--bands N]`、`sparse-regions <run> top-k` | 时间带占用表（busy_cores/total_cores、任务数）、稀疏段排行及分类 |
| **Z2 区域解释** | 一个时间窗里发生了什么、为何如此 | `region <run> --t0 .. --t1 ..`、`why-sparse <run> --band i`、`core <run> --core c` | 窗内活动任务、空闲核及其确定性原因、该带与上下游的时间关系 |
| **Z3 算子与依赖** | 单个算子的身份、时序与依赖 | `task <run> <id>`、`deps <run> <id> [--dir in/out]`、`subgraph <run> <id>` | 任务全时序、args/tensor 元数据、进出边、BFS 邻域 |
| **Z4 微观归因** | 算子内部的等待到底花在哪 | `why-late <run> <id>`、`why-long <run> <id>`、`rows <run> <id>`、`scheduler <run> --around <id>`、`early-dispatch <run> <id>`、`pmu <run> <id>` | 行级时序、stall 分解、调度/编排相位、early-dispatch 证明、PMU 比值 |

纵向过滤轴（可叠加于任意层）：`--family`、`--engine`、`--core`、`--t0/--t1`、
`--rank`（多 rank 库）。层级之间靠坐标导航，而不是让 agent 重新扫描。

### 6.3 `why` 查询与确定性推导规则

`why` 查询只输出**可证明**的解释；证明规则固化在衍生层并写入测试：

**`why-sparse`（某带某引擎占用低）**，按优先级判定：

1. 若带内存在"已就绪但未调度"任务（所有 pred 的 FIN ≤ t 而 start > t）→
   `kind=dispatch_wait`（调度/下发等待），引用该带对应的 scheduler 相位；
2. 若无就绪任务 → `kind=ready_starved`（上游没喂够）：列出仍活动任务中
   **最晚结束的 pred** 及其 FIN 时刻，说明"卡在上游 X 的到期时间"；
3. 该带属于 `drain_tail` 后缀 → `kind=drain_tail`（收尾波浪），附统计说明；
4. 判不出 → `kind=unknown`，证据 `unproven`，**绝不**给"带宽不足"等猜测。

**`why-late`（为什么不能更早启动）**，对任务计算：

- `ready = max(FIN(pred))`（pred 取 deps 边全集，口径见 5.3，T1 对拍锁定）；
- `fin_detect = dispatch - ready`（调度器发现就绪的延迟）；
- `dispatch_wait = receive - dispatch`（下发到手时间，AICPU 队列证据）；
- `start_wait = start - receive`（到手但未上核；若该核此区间被他人 task_row
  覆盖 → `measured`，否则原因未证 → `unproven`）；
- 外赠拓扑上下文：上游链长度（为什么它天然晚）。

早期派发事实：生产者 `allow_early_resolve` 标志 + 时间戳两 tick 容差证明，
输出 `full/partial/none/unavailable`（沿用运行时语义规则）。

**`why-long`（为什么跑得久）**：`busy_us` 与同 run 同 family 的中位数的
序位、行数/行时长偏斜、若有对齐 PMU 给管道占用比值。只陈述**测量**，
不做归因。

### 6.4 示例会话（O = Z0→…→Z4 导航路径）

```text
> pfdb query runs list
RUN run_id=7 program=Qwen3Decode platform=a2a3 captured_at=... bench_mean_us=...
> pfdb query overview 7
METRIC run_id=7 makespan_us=1868.6 ...         # 顶线
RESOURCE run_id=7 engine=aic cores=20 ...      # 捕获实际拓扑，来自 metadata
RESOURCE run_id=7 engine=aiv cores=40 ...
> pfdb query density 7 --engine aiv --bands 20
BAND band_idx=9  busy_cores=1  total_cores=40 ...   # ← 稀疏带（本捕获 48% 的带为空）
BAND band_idx=10 busy_cores=32 total_cores=40 ...   #    相邻带却接近满载（双峰结构）
> pfdb query why-sparse 7 --band 9
SPARSE run_id=7 band=9 engine=aiv kind=ready_starved evidence=proven
  lagging_producer=qk_matmul fin_us=...             # ← 上游最晚到期
> pfdb query task 7 <softmax-id>
TASK run_id=7 task_id=... name=softmax family=softmax engine=aiv busy_us=...
> pfdb query deps 7 <softmax-id> --dir in
DEP pred=qk_matmul succ=softmax tensor_edges=1 tensor={dtype, shape, strides}
> pfdb query why-late 7 <softmax-id>
STALL ready_us=... dispatch_us=... receive_us=... start_us=...
  fin_detect_us=... dispatch_wait_us=... start_wait_us=... evidence=measured
```

### 6.5 横向（对比）查询

| 查询 | 作用 |
|------|------|
| `compare <run_a> <run_b>` | 兼容性门禁 + 中性 before/after 差值/比值（沿用采集方兼容性口径） |
| `baseline list` / `baseline diff <run>` | 相对基线的回归检查（对比是旧数据的唯一长期价值） |
| `trial <id>` / `trials --active` | 当轮实验目标/假设/结论与血缘链 |

### 6.6 查询目录：按 kernel agent 的信息需求演进（机制，不是冻结清单）

**定位**：接口只冻结"信封"——Result 结构、事实 DSL、证据状态、字节预算、
坐标导航；不冻结"信件"——查询名录与参数。名录是**活目录**，以 kernel agent
的真实信息需求为唯一生长依据，在开发与使用过程中补充、裁剪；本文列出的
种子查询不视为最终契约，现在的空缺由后续开发补齐。

**演进机制**（T4 起生效）：

1. 每个查询在 `query/registry.py` 注册，强制携带 **owner question**（"agent
   用它回答什么问题"）；CLI/MCP 入口由注册表自动生成（pydantic 单一同源）。
2. 新增查询 = 注册模块 + 至少一条金质题 + 文档目录表一行，**不动核心**；
   新增是向后兼容（minor）；改名/删除走 deprecate → 裁剪（本系统数据可弃、
   消费方单一，裁剪成本可控）。
3. 想做的查询先进"候选池"；没有绑定 owner question 的查询拒绝实现——
   杀掉"我猜 agent 可能需要"式投机功能。
4. agent 落地后对工具调用做计数埋点（预留）：长期无人用的查询淘汰出目录；
   目录瘦身与生长同等重要。

**agent 信息需求框架（种子目录的出处）**：站在 kernel agent 的优化循环上，
把"为了实现性能优化，我还需要知道什么"拆成七个阶段：

| 优化循环阶段 | agent 的典型问题 | 对应的种子查询群 |
|--------------|------------------|------------------|
| **定向**（我改什么、从哪出发） | 基线是多少？平台/形状/配置/版本是什么？最近可用的 run 是哪个？ | `runs list`、`overview`、`baseline list` |
| **全貌**（时间去哪了） | makespan 里 compute 与 stall 各占多少？哪个引擎闲？哪些 family 最贵？图多深多宽、并发放得开吗？ | `density`、`sparse-regions`、family 聚合、CPM 顶线 |
| **定位**（先打哪里） | 真正决定时长的链是哪条（observed/static）？stall 在路径上哪几个任务炸开？哪些时间带利用率塌陷？ | `critical-path`、`region`、`why-sparse` 目标筛选 |
| **归因**（为什么慢/为什么空） | 这个任务为什么不能早启动，FIN→dispatch→receive→start 各段各花多少？这段空窗是没人可派还是调度没派？early dispatch 的资格用上了吗？ | `why-late`、`why-sparse`、`why-long`、`early-dispatch`、`scheduler --around` |
| **施动前的约束**（动手前必须知道什么） | 候选算子的直接依赖是谁、边上的张量形状/stride/dtype？缓冲区离上限多远？编译器给了什么 tile hint？PMU 里哪根管子接近满载？ | `deps`、`subgraph`、`memory`、`perf-hints`、`pmu` |
| **验证**（改动生效了吗） | bench 动了多少？profiled 指标怎么联动？对比的两个 capture 前提一致吗（同配置/形状/工具链）？ | `compare`、`baseline diff`（内置兼容门禁） |
| **记忆**（沉淀与再出发） | 这个假设之前试过吗？上次同类改动是 win 还是 regression？基线要更新吗？ | `trial`、`trials --active`、`baseline add` |

两条边界（与非目标呼应）：

- "**怎么改**"（具体优化策略）不提供——那是 agent 的职责；本库只保证
  施动前的约束信息与施动后的验证证据是齐的。
- 数值正确性验证属于 golden 校验域，本库不覆盖。

**候选池**（当前空置，出现对应 owner question 才转正）：family 瓶颈综合
排行、引擎失衡诊断、尾段（drain tail）专项、跨 run 占位对比、scheduler
等待带热点描述、渲染图摘要文本化……池子开放，问题驱动进出。

---

## 7. 视觉渲染层

对多模态 LLM 与人类抽查的第二通道；**渲染是派生视图，原始 JSON 永远不进上下文**。

- **四级渲染**：
  - `R0 whole`：全部核 × 全时间轴（µs），低分辨率总览；
  - `R1 window`：指定时间窗（±上下文），横轴放大，附依赖箭头与 pred FIN 标线；
  - `R2 task`：单算子邻域：本任务行高亮 + 直接生产者/消费者 + 就绪线；
  - `R3 core`：单核时间轴（回答"这核空着的时候别人在干嘛"）。
- **确定性**：样式参数集中在一个常量模块；同 `(run, kind, params)` 渲染结果
  SHA-256 稳定（Python/matplotlib 版本记录在清单里）。
- **缓存与清单**：`render/<run>/<kind>-<params_hash>.png` + 同名
  `render_manifest.json`（宽高、µs/px、图例、生成版本）。缓存随库作废
  （prune 连带清理），重复请求命中缓存。
- **预算**：图像受尺寸/字节上限约束，超限自动降采样并在 manifest 标注。
- 文本模型路径：查询返回 `IMAGE` fact（清单元数据），模型可忽略像素只用
  语义事实；多模态模型请求实际图像。

---

## 8. 生命周期与短期记忆（时效性管理）

### 8.1 工作集与清理策略

- **工作集** = 最新 `K` 个 run（默认 `K=3`）+ 所有 baseline 引用的 run +
  所有 active trial 引用的 run。保留行打 `run.retained=true`。
- **prune 策略**：`pfdb prune --keep 3`（默认在每次 ingest 成功后自动执行，
  `--no-prune` 可关）：
  - 非保留 run：删除其全部表行（run/artifact/task/task_row/.../衍生行），
    link 模式不删任何文件，copy 模式连带删除 store 副本与渲染缓存；
  - trial/baseline 行本身极小，永久保留（是短期记忆的精髓）；
  - 若某历史 run 突然需要对比价值，`pfdb baseline add <run>` 先标记再 prune。
- **体积控制**：原始工件默认 link 不复制；in-core/args_dump 永远只存指标；
  渲染缓存有总量上限（默认 200MB，LRU）。
- **重建**：`.pfdb` 删除后，用 `pfdb ingest build_output/... --replay manifests`
  或原命令重跑 ingest 即可重建相同内容的库（sha256 保证一致性）。

### 8.2 短期记忆（trial / baseline）

- **trial 生命周期**：`register_trial(goal, hypothesis, changed_files)` →
  ingest 绑定 `run_id` → `set_verdict(win/neutral/regression, evidence_refs)`。
  血缘树可回溯（`parent_trial_id`），形成"假设 → 实验 → 证据 → 结论"链条，
  服务于**当前优化循环**；循环结束由 agent 决定是否归档（导出 report 或
  仅留 baseline）。
- **baseline**：命名 + `bench_mean_us`（以未 profiled 的 `PYPTO_BENCH` 为准；
  profiled makespan ≠ 基准）+ 验收标准；`baseline diff` 做相对基线变化，
  需 level/时钟/拓扑/程序名兼容（沿用采集方口径）。
- **不承诺长期历史**：`history --family` 等跨 run 聚合只覆盖保留 run；
  长期知识沉淀不在本系统范围。

---

## 9. 接口层

- **Python API**（唯一事实源）：
  `profile_db.api.ProfileDB(path)`：`ingest(source, **meta)`、`query(name, **kw)`、
  `render(...)`、`prune(...)`、`trial/...`；返回统一
  `Result(facts: list[Fact], images: list[ImageRef], truncated: bool)`。
  查询参数 pydantic 严格校验；支持 `ProfileDB.memory()` 纯内存模式。
- **CLI**（`pfdb`）：`pfdb init|ingest|list|query|compare|render|prune|
  note|baseline|trial|...`，输出 `facts|json|markdown` 三种格式。CLI 与
  API 走同一引擎（输出逐字节一致，纳入测试）。
- **MCP 服务**（agent 主通道）：`pfdb serve --mcp`（stdio）。工具集 =
  查询注册表自动生成的**种子目录**（当前含 `pfdb.list_runs / overview /
  density / sparse_regions / region / task / deps / why_late / why_sparse /
  compare / baseline_diff / render / register_trial / note` 等，随 6.6 机制
  演进）。工具 JSON Schema 与 Python API 参数定义**单一同源**
  （pydantic 出 schema），避免多份定义漂移。MCP 返回 Result envelope；
  图像以 MCP 图像内容类型返回。
- **进程生命周期**：MCP 服务**不常驻**。kernel agent 执行调优工作时以子进程
  启动 `pfdb serve --mcp`（stdio，随 agent 会话同生命周期），结束即退出。
  本任务只交付稳定的服务入口与 tool 契约，消费方 agent 的实现不在本任务范围。
  接口稳定性保证：tool schema 版本号随设置发布，破坏性变更走语义化版本。
- 参考现存 `agents/openai.yaml` 的接入习惯，提供同类示例配置。

---

## 10. 仓库布局与依赖规则

### 10.1 目录

```text
profile_db/
├── DESIGN.md                 # 本文档
├── README.md                 # T0 生成
├── pyproject.toml            # 包名 profile_db，CLI 入口 pfdb
├── src/profile_db/
│   ├── db.py                 # 连接、迁移、写锁
│   ├── schema/migrations/    # NNNN_*.sql（版本化 DDL）
│   ├── ingest/               # 适配器注册表 + 各工件插件（ingest/adapters/*.py）
│   ├── derived/              # 衍生器：band/gap/cpm/stall/early_dispatch（纯函数）
│   ├── query/                # Z0–Z4 + why + 对比查询；facts DSL v2 输出
│   ├── lifecycle/            # 工作集判定、prune、重建
│   ├── render/               # R0–R3 渲染器、样式常量、缓存、manifest
│   ├── api.py                # ProfileDB 公开 API + Result 模型
│   ├── cli.py                # pfdb 命令行
│   └── mcp_server.py         # MCP tools（stdio，按需启动）
└── tests/
    ├── fixtures/             # 合成工件生成器 + 小型金质工件
    ├── golden_qa/            # 金质题库（10–15 题及其期望 facts）
    └── test_*.py             # 单元/集成/快照测试
tools/（仓库根目录，与 profile_db/ 平行）
└── calibrate.py              # 标定工具（由原 scratch 探针收编）
```

运行时数据（git 忽略）：`.pfdb/profile.duckdb`、`.pfdb/store/`（copy 模式）、
`.pfdb/render/`（渲染缓存）。

### 10.2 依赖规则（可维护性的硬约束）

```text
query ──▶ 只依赖 schema 表与视图（不 import ingest/解析器）
render ──▶ 只依赖 schema 表
ingest ──▶ 只写库，可依赖 derived（入库后触发衍生）
derived ──▶ 纯函数：输入=表/流，输出=表；不碰文件系统原始 JSON
lifecycle ─▶ 只依赖 schema 表（判定保留集）＋ filesystem（删除副本）
mcp/cli ──▶ 只依赖 api
tests ────▶ 可直接触达任何层，但金质题库只走 api/CLI/MCP 三个入口
```

违规由 import-linter 类检查在 CI 拦截（T0 接线，T5 收紧）。

---

## 11. 子任务分解（里程碑 T0–T10）

> 每个任务：目标 / 做什么（明确"不做什么"）/ 交付物 / 验收标准（可自动验证）/ 依赖。
> 规则：验收必须离线可执行；每里程碑结束留下可用增量；可维护性要求见 10.2。

### T0 骨架与契约（含完整 schema v1） ｜ 依赖：无 ｜ 规模：S

- **做什么**：建包 `profile_db/`；连接管理与写锁；`migrations/0001_init.sql`
  一次落全部 5.2 表（含衍生/知识层表）；事实 DSL v2 与证据状态枚举；
  合成工件生成器 fixtures（最小合法 records/deps/name_map，全离线测试用）；
  将标定探针收编为 `tools/calibrate.py`；CI 接线
  （pytest + ruff + 头检查 + import 规则检查）；`.gitignore` 增加 `.pfdb/`；
  README。（**T0 已完成**，见 README 状态表。）
- **不做什么**：不接任何真实工件解析。
- **交付物**：可 pip install -e 的包（Python ≥3.10，pyproject 声明依赖）；
  `pfdb init` 可建库。
- **验收**：
  - [ ] `pytest` 离线全绿；新建/重复打开/升级 DB 幂等，`schema_version` 正确；
  - [ ] 合成 fixture 生成器产物通过 schema 校验工具；
  - [ ] 证据枚举不可构造非法值（类型层面报错）；
  - [ ] lint 与 license 头检查通过；`.gitignore` 生效（git check-ignore 断言）。

### T1 摄取：L2 泳道族 ｜ 依赖：T0 ｜ 规模：M

- **做什么**：适配器 `chip_swimlane_records.json`（及 `l2_swimlane_records` /
  `l2_perf_records` 兼容名归一）+ `deps.json`（边原始字段保真、逐条入库）+
  `name_map*.json` + merged trace 清单；时钟域合并复用
  `simpler_setup.tools.swimlane_converter.read_perf_data`；`store_mode`
  默认 link（sha256 照算、文件不复制）；ingest 幂等（同 run 重复 ingest
  结果稳定；事务保护，失败不留半库）；run 环境元数据采集（git commit/dirty、
  runtime_cfg、cmdline，脱敏）。（**T1 已完成**，见 README 状态表。）
- **不做什么**：不做 PMU/in-core；不做查询接口；不复制文件（link 默认）。
- **验收**：
  - [x] 对真实 Qwen3Decode 捕获：task=266、task_row=706、dep_edge=2546，
        artifact 表 4 个条目 sha256 正确、store_mode 全为 link、无文件复制；
  - [x] 同 run 重复 ingest 两次，任何表行数不变（幂等）；
  - [x] 抽样 5 个 task 的 µs 时序与 `read_perf_data` 直出对拍（误差 0）；
  - [x] 截断/损坏 JSON、缺失 name_map：报结构化错误，事务回滚，库保持一致；
  - [x] `chip_swimlane` 与 `l2_swimlane` 两种命名摄取结果一致（对拍测试）；
  - [x] makespan/busy/wall 口径与 5.3 决策表一致（数值断言）。
- **实施备注**：`scheduler_phase.shared_at_*` 实证为每队列深度列表（v1 误标
  INTEGER），以迁移 `0002` 修正为 JSON；真实捕获 makespan=2828.500µs
  （converter 口径），相位入表 746 条调度 + 365 条编排记录。

### T2 摄取：文本类证据 ｜ 依赖：T0（可与 T1 并行） ｜ 规模：S

- **做什么**：`perf_hints.log`（origin=compiler 原样）；memory report 解析；
  `pmu.csv` 动态列名长表化；基准数据注册（解析 `PYPTO_BENCH` 日志行或
  `pfdb ingest --bench 'mean=... min=...'`）。（**T2 已完成**；run 环境元数据
  （git commit/dirty）随 T1 落地，runtime_cfg 入库时做路径脱敏。）
- **验收**：[ ]→[x] 三类工件各 1 个 fixture 全量对拍解析结果；[x] pmu 两套
  不同列名 fixture 均正确入长表；[x] 缺省工件时 ingest 成功且对应查询返回
  `unavailable`；[x] 环境元数据不含机器用户名（测试断言，覆盖
  `/data1/home/<user>` 云主机布局）。
- **实施备注**：真实捕获 `report/perf_hints.log` 62 行全部保真入库
  （含编译器原始绝对路径，origin=compiler 标识）；bench 数字在重摄取时
  按"显式提供者胜、缺失者继承"合并。

### T3 衍生层 ｜ 依赖：T1 ｜ 规模：M

- **做什么**：`time_band` 密度索引（5µs 存储粒度、自适应上限 10k 带、
  sparse/drain_tail 判定按 5.3）；`idle_gap` 分类（≥5µs 记录）；CPM
  observed/static 落 `cpm_path`；stall 分解四段；early-dispatch 证明。
  全部纯函数、幂等、带证据标注；输入输出约束检查（核数/时钟从工件读，
  不硬编码）。（**T3 已完成**，见 README 状态表。）
- **不做什么**：不写查询输出格式。
- **验收**：
  - [x] 对真实捕获：CPM 任务序列与 `python -m simpler_setup.tools.critical_path`
        输出一致（对拍）；stall 分解和 = gap（一致性断言）；
  - [x] 每个 `idle_gap.kind` 至少一个构造场景单测，`evidence` 标注正确；
  - [x] 稀疏判定单测：复制真实 AIV 双峰分布（48% 空带）的合成数据，
        判定结果与 5.3 阈值一致；drain_tail 形态规则单测；
  - [x] 重复运行衍生器结果逐字段一致；空输入/单任务/零边边界不崩、
        标注 unknown/unavailable。
- **实施备注**：`derived/` 为纯函数子包（输入=表、输出=行清单，不碰原始
  JSON），ingest 事务内触发入库；`time_band/idle_gap/cpm_path` 与
  `run.cpm_us`、`task.on_cpm_*` 随 run 一并重建，重摄取幂等。真实捕获对拍：
  static 路径 12 任务 / observed 20 任务序列、gap kind、逐任务
  compute/stall 与上游零差（fp 噪声 ≤3e-13µs），static CPM 2405.98µs；
  密度带复现标定数字（aic 566 带/空 20/稀疏 37/drain 5，aiv 566 带/空 270/
  稀疏 364/drain 5 —— 附录 B 的 3.5% 与 47.7% 空带精确吻合）；idle_gap
  340 条（dispatch_wait 68 / ready_starved 269 / unknown 3，drain_tail 在
  该捕获为 0）；level-1 边界按"FIN 流不可用"语义处理（绝不解释 0.0
  占位符）。

### T4 分层查询引擎（核心，最大的里程碑） ｜ 依赖：T1、T3 ｜ 规模：L

- **做什么**：Z0–Z4 全部查询 + `why-sparse/why-late/why-long` + 纵横过滤轴 +
  DSL v2 输出 + 字节预算与 `TRUNCATED`；查询以注册表组织、每个查询绑定
  owner question（6.6），CLI/MCP 入口由注册表自动生成；**无原始 JSON 泄漏
  检查器**（输出中任何 JSON 片段必须来自 schema 表字段，测试兜底）。
  （**T4 已完成**，见 README 状态表。）
- **验收**：
  - [x] 金质题库 10–15 题（含 6.4 全会话）全部通过（snapshot 断言期望 facts）；
  - [x] 每条输出事实恒带证据状态（模式级校验）；
  - [x] 超预算输出以 `TRUNCATED` 收尾且不静默丢行；
  - [x] 不存在的 task/band 返回 `unavailable`，不发生推测；
  - [x] 多 rank 库：不带 `--rank` 的确定性查询拒绝回答并列出候选；
  - [x] 查询注册表自检：每条注册查询必须携带 owner question（6.6）并绑定
        至少一条金质题，缺失即 CI 失败。
- **实施备注**：`query/` 包 = 注册表（`registry.py`：QuerySpec 强约束
  name+owner question+pydantic 参数模型，`execute` 统一解析/校验/渲染）+
  Z0–Z4 共 17 条查询（handlers_z0..z4）。处理器只读 schema 表与 derived
  纯函数（stall/early_dispatch 复用，绝无原始 JSON），事实 DSL v2 输出、
  `us()` 显示舍入到纳秒、字节预算 `TRUNCATED remaining=.. limit=..` 显式收尾；
  不存在的 run/task/band 一律回 `unavailable` 事实、不猜测。多 rank 守门落在
  `runs_list`（存在非 `single` rank 且未指定 `rank` 时抛 `QueryError` 并列候选）。
  无原始 JSON 泄漏检查器（`tests/golden_qa/json_leak.py`）断言每个 list/object
  值要么等于某 schema JSON 单元、要么元素全为该 run 的真实 task id。金质题库
  20 题离线快照（含 6.4 全会话逐步骤）+ 真实捕获 4 题锚点（skipif）；
  全量 153 测试通过。真实捕获锚点：overview makespan 2828.5µs / static CPM
  2405.98µs / 266 任务 / aiv 密度 20 显示桶（566 存储带）。

### T5 接口：Python API + CLI ｜ 依赖：T4 ｜ 规模：S

- **做什么**：公开 API 定型（Result 模型 + pydantic 参数校验 +
  `ProfileDB.memory()` 内存模式）；`pfdb` CLI 全查询子命令与
  `facts/json/markdown` 格式化；`agents/openai.yaml` 风格接入示例。
- **验收**：[ ] CLI 与 API 输出逐字节一致；[ ] `pfdb --help` 与 README 示例
  全部可执行；[ ] 参数非法时报结构化错误（快照）；[ ] 内存模式行为与磁盘
  模式一致（同 fixture 对拍）。

### T6 可视化渲染层 ｜ 依赖：T3（可与 T4/T5 并行） ｜ 规模：M

- **做什么**：R0–R3 渲染器 + 确定性样式常量 + 缓存（总量上限 LRU）+
  `render_manifest.json` + 尺寸/字节预算与降采样。
- **验收**：[ ] 同参数两次渲染 SHA-256 一致；[ ] 窗口渲染 x 轴范围与来源表
  区间对拍；[ ] manifest 字段完整；[ ] 空窗口/无边任务渲染出带说明的图或
  显式 unavailable，不崩；[ ] 缓存上限生效（写入超量后旧缓存被逐出）。

### T7 MCP 服务 ｜ 依赖：T5（图像工具依赖 T6） ｜ 规模：M

- **做什么**：`pfdb serve --mcp`（stdio，按需启动、会话级生命周期）；tools 与
  API 参数同源；Result envelope；图像以 MCP 图像内容返回；附 mock-agent
  脚本演示完整泳道阅读会话；tool schema 版本化。
- **验收**：[ ] mock 脚本仅用 MCP 工具完成"整体→稀疏→原因→算子→依赖→
  为什么晚"全会话，每步输出在预算内；[ ] 非法参数被 MCP 层拒绝并返回
  可用错误信息；[ ] 服务可重复启停（无状态残留，第二次启动结果一致）。

### T8 生命周期与短期记忆 ｜ 依赖：T1（可与 T4–T7 并行） ｜ 规模：M

- **做什么**：工作集判定（K=3 + baseline/active trial 引用保护）；`pfdb prune`
  与 ingest 后自动 prune（默认开、`--no-prune` 关闭）；trial 注册/结论/
  血缘回溯；baseline 管理与 `baseline diff`；compare 兼容性门禁。
- **验收**：[ ] 构造 5-run 场景，auto-prune 后仅剩最新 3 run + 被 baseline
  引用的旧 run，行数/文件删除精确断言；[ ] link 模式 prune 不触碰任何文件、
  copy 模式连带删除；[ ] 模拟 3-run 调优循环端到端走通；[ ] 不兼容对比被拒
  并说明原因；[ ] 重建测试：删库 → 重 ingest → 查询结果一致（sha256 保障）。

### T9 扩展模态（可延后） ｜ 依赖：T1/T2 ｜ 规模：L

- **做什么**：in-core（manifest 状态、instr 指标入表、原始 trace 永不复制）、
  args_dump 元数据、scope_stats 元数据。
- **验收**：每类 1 个 fixture 对拍；原始大文件确认不进关系表也不进 store
  （体积断言）。

### T10 文档、技能与定型 ｜ 依赖：任意（最后做） ｜ 规模：S

- **做什么**：`docs/debug-and-tune/` 用户手册（与代码行为一致）；决定现有
  profile-feedback skill 与 DB 的并存/替代关系并落地；发布检查清单。
- **验收**：[ ] 文档示例全部可执行；[ ] 仓库 lint 全绿；[ ] 技能决策有记录。

### 里程碑依赖图

```text
T0 ──▶ T1 ──▶ T3 ──▶ T4 ──▶ T5 ──▶ T7
  └─▶ T2 ─┘   └─▶ T6 ────────────┘ ↑
  └──────▶ T8（可与 T4–T7 并行）
T9（T1/T2 后随时启动）    T10（最后）
```

---

## 12. 风险与开放问题

| 风险/问题 | 应对 |
|-----------|------|
| 时钟域合并复杂（AICore/AICPU 两套时钟） | 只复用 `read_perf_data`，自研拼接被测试禁止 |
| 运行时工件命名漂移（chip/l2/legacy） | T1 归一化适配器 + 两侧对拍测试 |
| 拓扑差异（20/40 vs 24/48 等） | 一切拓扑/时钟从 artifact 读，硬编码被 lint/测试禁止 |
| prune 误删有价值 run | baseline/active trial 引用保护 + 保留标记显式化；重建兜底 |
| `build_output` 被清理后无法重建 | 库仍可查（相对路径失效时对应查询返回 unavailable）；副本需求显式用 copy 模式 |
| DuckDB 单写锁竞争 | ingest 集中唯一写者入口；查询只读连接 |
| in-core/args_dump 体积大 | 原始文件永不复制，只存指标/元数据 |
| profiled makespan 观察者开销 | bench 与 makespan 语义分离（5.3），文档强调 |
| 衍生层过度演绎（"稀疏=带宽不足"） | 证明规则白名单（6.3），其余一律 unproven |
| 渲染平台差异（字体/版本） | 样式常量 + 版本记录 + snapshot 测试 |
| 与现有 skill/文档的过渡 | T10 显式决策并存或替代，保持 docs 恒真 |

**残留待定项**：
1. `dep_edge` 的 `flags` 字段语义（artifact 边与否对 `ready` 计算的影响）
   以 simpler 运行时为准，T1 用 critical_path 工具对拍锁定——这是唯一
   需要上游确认的口径点；
2. `history --family` 在"不承诺长期历史"定位下是否保留：默认做简版
   （仅保留 run 内），T8 讨论去留；
3. 渲染样式（配色/图例）属实现迭代项，不阻塞主线；
4. 查询目录（6.6）的第一轮收敛以 T4 金质题库为界，后续以 kernel agent
   的实际使用为准增删——这是"接口先不定死"的落地机制。

---

## 附录 A：与 profile_feedback.py 的关系

- 仅吸收两个概念：**证据状态**（measured/proven/unproven/unavailable）与
  **字节预算 + TRUNCATED 显式标记**。
- 不迁移代码、不要求 CLI/输出兼容；该脚本继续独立存在，试点期过后由
  T10 决定去留。

## 附录 B：标定数据（`tools/calibrate.py` 对 Qwen3Decode 真实捕获实测）

```text
捕获: build_output/Qwen3Decode_20260825_101508/dfx_outputs
拓扑: 60 核 = 20 AIC + 40 AIV（artifact metadata），时钟 50 MHz
规模: aicore 物理行 706，原始行跨度 2825.18 µs；deps: 266 任务 / 2546 边

任务 busy 时长:  AIC  min 14.5 / p50 246.0 / p90 597.3 / max 984.2 µs
                 AIV  min 1.8 / p50 11.9 / p90 594.3 / max 621.7 µs
同核行间间隔:    AIC  p50 0.94 µs（>1µs 占 45.2%）
                 AIV  p50 40.88 µs（>5µs 占 97.8%，max 1656 µs）
5µs 带占用:      AIC  p50 20/20，空带 3.5%，≤25% 核数 6.9%（近全天候满载）
                 AIV  p50 1/40，空带 47.7%，≤25% 核数 64.7%，p90 32/40（双峰）
```

由以上数据锁定 5.3 决策：存储粒度 5µs（自适应上限 10k 带）、稀疏阈值
≤25% 核数、idle_gap 记录阈值 5µs、显著间隔标记 1µs、拓扑全从工件读。