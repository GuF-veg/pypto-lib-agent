# 面向算子开发 Agent 的性能反馈：从两个算子实践重新审视 pfdb

研究日期：2026-10-21。状态：**研究结论与待验证的设计建议，不是已批准的实现规格**。

本研究只读取已有代码、文档、工件和数据库，补充查阅第一方研究。未修改 pfdb、算子、runtime、测试或原始采集；未重新跑设备 benchmark。文中的算子性能来自 2026-10-09 / 10-14 的两次研究，接口回放与代码审查发生于本研究日期。

## 1. 核心结论

**pfdb 下一步最值得优化的是证据之间的连接及其面向决策的表达。继续扩大 profiling 指标覆盖面，优先级低于帮助 agent 判断“这一份证据对应哪段计算、支持哪种解释、在什么条件下可以比较、下一次实验应验证什么”。**

两次实践不支持“agent 不会融合”“agent 不会用核”“手写一定更快”这样的统一解释。它们呈现的是不同实现选择的组合：SWA 的 agent 输在任务提交组织、热点 primitive 和部分依赖粒度；compressor 的 agent 在已测连续 fixture 上，凭更便宜的候选定位和紧凑向量路径胜过手写版，但投影仍较弱，RoPE 仍有可优化的慢指令路径。

由此建议保留 pfdb 的离线数据库定位，并将改进重点放在六件事上：

1. **先绑定比较条件。** 工作负载、输入语义、数值验证和计时目标，是性能证据的一部分，不能只靠相同 program 名称。
2. **以计算阶段连接 task、block、PMU 和代码。** 同名 family 比较不足以处理融合、拆分、改名和跨 chunk 执行。
3. **解释任务内部的时间几何。** task 跨度、每个 block 执行时间、分散启动和累计 core-time 应明确区分。
4. **补上源码到实际实现的桥梁。** profiling 指出 scalar 高；代码/IR 才能区分 512-row 扫描、scatter 标量化、同步等待或重复计算。
5. **把差分实验及其适用条件结构化。** 记录改了什么、保持什么、预期看到什么、正确性是否通过，保留失败与无收益结果。
6. **默认输出少量相关证据，完整细节可达。** 压缩默认响应与工具发现成本，同时保证能够分页、过滤、定位源码和检索完整证据；不能只靠截断。

这不是将 pfdb 改成自动优化器。**runtime 继续定义运行时分析口径，pfdb 负责证据归一化、关联和访问，算子开发 agent 负责解释、提出候选和执行实验，外部 harness 负责采集与验证。**

## 2. 研究依据、方法与推断边界

### 2.1 本地依据

| 依据 | 本次使用方式 | 可以支持的结论 |
| --- | --- | --- |
| [SWA 性能研究](../../benchmarks/implementation/deepseek-v4.1-flash/performance-gap-study.md)及 [12 组配置 CSV](../../benchmarks/implementation/deepseek-v4.1-flash/research_assets/swa_gap_measurements.csv) | 对照实验、反例、源码与生成代码线索 | 特定实现选择对已测 workload 的影响 |
| [Compressor 性能研究](../../benchmarks/implementation/deepseek-v4.1-flash/compressor-ratio4-performance-gap-study.md)及 [11 组配置 CSV](../../benchmarks/implementation/deepseek-v4.1-flash/research_assets/compressor_ratio4_measurements.csv) | 相反性能排序、数值限制、长序列行为 | 已验证 fixture 上的优势和机制证据 |
| `profile_db/` 当前实现、schema、CLI/MCP、测试 | 从公开入口追踪到真实 handler 与数据来源 | 当前已经提供、难以使用或尚未建模的能力 |
| 现有 compressor 数据库的 7 个 run | 只读回放、响应规模、失败路径、比较门槛 | 当前接口的具体行为，而非臆测使用困难 |
| [runtime 集成台账](pfdb-runtime-integration.md) | 核对权威口径、复用工具、兼容与版本锚定 | 后续设计应复用的现有基础 |

代码审查锚定 pypto-lib `d6cbf967bcf0d4b4bc412467cf28c944e26b1362`，pfdb 0.4.0 / schema 8 / MCP schema 4。四个原始算子文件的 SHA256 仍与两份报告一致。只读回放前后检查了所打开数据库的文件 hash；没有修改数据库。

本次没有依赖旧顶层设计稿中的历史描述来判断现状：[DESIGN.md](../../profile_db/DESIGN.md) 已明确标记部分旧口径被替代；以下以当前源码和集成台账为准。旧 [dogfooding 报告](profile-db-dogfood-feedback.md) 中已修复的 task-ID、rank、modality provenance 问题，不重新列为缺陷。

### 2.2 不能从两次实践推出什么

- 没有原始开发 agent 的完整可观察轨迹：读过哪些材料、收到哪些工具响应、试过什么候选、预算在哪里耗尽。因此不能断言它“没有硬件知识”“忽略了 PMU”或“因为工具太多而失败”。
- 两类算子的 23 个有效配置不是 23 个独立算子样本；不能据此估计所有 agent 的能力分布。
- compressor 的 agent 优势限于三个通过相同 golden 的单请求连续 fixture；clip、分母 EPS、physical locality 等差异尚未完成通用语义等价验证。
- 历史 benchmark 每个配置一个进程、100 轮、5 warmups，没有跨独立进程的置信区间。0.8–2 µs 级差异只能作为本次观察，不能升级为稳定规律。
- 本次没有做“旧 pfdb / 新接口”的算子优化 A/B 实验。后文给出的是有证据支持的设计假设及验证方法，不是已经证明的新系统收益。

**证据强度应区分：已测结果、源码/图结构证明、基于案例的设计推断、尚待实验的系统假设。**

## 3. 两个算子究竟教会了我们什么

### 3.1 经验矩阵

下表中的时间均沿用原报告的无插桩中位数；微基准单独标明，不能与完整算子差值相加。

| 案例与干预 | 已测结果 | 形成有效假设需要什么 | 单独看什么容易误判 |
| --- | --- | --- | --- |
| SWA：32 次 attention 提交合并为一个 SPMD grid | 274.5→250.4 µs；tasks 51→20，物理行仍 178 | 实际 submit 数、grid/block 区分、生成编排和图结构 | Python 写了 `pl.parallel`，或“块已经够多” |
| SWA：RoPE scatter/gather 微基准 | 33.6→10.8 µs，零容差比较通过 | 热点定位、具体 axis/mask/dtype、生成代码、pinned ISA 实现 | primitive 名称；把微基准差值乘调用数预测整算子收益 |
| SWA：手写前段 + agent 原投影 | 274.5→213.2 µs | 跨融合/拆分结构的阶段对应，确认哪些计算体保持不变 | 把联合替换的收益全部归给某一项技巧 |
| SWA：B 投影等待所有组改为等待本组 | 261.2→247.4 µs | producer-consumer 窄边、tensor region、数学组独立性、保持相同 kernel/grid/提交顺序 | 只数 task/edge，或泛化为“删除依赖” |
| SWA：没有形成组间流水的投影拆分 | 250.4→257.0 / 265.3 µs | 新增开销与实际重叠窗口的对照 | “拆得更细”“纯 cube/vector 一定更好” |
| SWA：请求 stage=2，实际只能容纳一份 buffer | 编译器提示 storage alias 与序列化；未独立量化 µs | realized buffer 数、地址/别名、容量、pipeline hint | 看见 `pl.pipeline(stage=2)` 就认为流水已经实现 |
| Compressor：agent 对手写 | 206.5 对 239.3 µs；agent 更快 | 同输入与 golden、阶段跨度、候选查找复杂度、PMU | “手写优化版”这个名称；从整算子排序推断每阶段排序 |
| Compressor：手写 pool 扫 512 行改为最多 8 个候选 | 239.3→184.5 µs；scalar busy 80.1%→27.2% | 源码循环边界、position/mapping 合法域、保持后续计算与 grid | 只看 pool 很长、scalar 高或 occupancy 高 |
| Compressor：agent RoPE 改 gather | 206.5→179.7 µs；scalar busy 47.0%→17.1% | task→生成函数→primitive 实现→真实设备对照 | “所有 scatter 都慢”的永久规则 |
| Compressor：关闭手写投影 K 流水 | 239.3→275.5 µs | tile/grid 不变的消融，实际缓冲和 L0 tile | 大 tile、Acc 占满就代表吞吐高 |
| Compressor：向量 blocks 40→128 | 206.5→208.3 µs；gather 下 179.7→182.1 µs | 单 block 工作量、波次、尾波、初始化和总 wall time | 把更多 blocks 当作单调优化方向 |
| Compressor：8,192 tokens 的投影交错与有序 pool | 第二 chunk 投影 task 跨度超过 1 ms，单 row 最长约 109 µs | chunk 对应关系、分散启动、完成顺序、状态依赖 | 把 task span 当作单个 matmul 的纯执行时间 |

实验与限制的完整出处分别见 [SWA 第 5–10 节](../../benchmarks/implementation/deepseek-v4.1-flash/performance-gap-study.md#5-原因一提交粒度带来的调度成本)和 [compressor 第 3–8 节](../../benchmarks/implementation/deepseek-v4.1-flash/compressor-ratio4-performance-gap-study.md#3-代码结构比较)。

### 3.2 agent 做对了什么

可以确认的是**产出的实现做对了什么**，而不是其未记录的内部推理：

1. compressor agent 在已测输入域中，用 bounded window 向量计算替代了完整 tile 的 metadata 候选扫描。即使向量算术存在冗余，整体仍更便宜。
2. 它把 pooling、RMSNorm、RoPE、cache write 组织为紧凑路径，减少了阶段边界；但没有保持其他因素相同的单独融合实验，不能给融合分配确定的 32.8 µs 收益。
3. SWA agent 也有正确的组内量化和并行结构；报告排除了“整个实现缺乏并行算法”或“同 family 必然存在错误 WAW 链”的笼统解释。

因此，手写实现也应当作为**有优缺点的候选与对照**，而非优化目标的唯一正确答案。pfdb 的输出应围绕当前 workload、执行事实和受控实验，不编码“追上手写”“复制专家结构”的默认立场。

### 3.3 为什么会错失优化：五种可能的断点

以下是从实现中重建的信息需求，不是对原始 agent 的心理归因：

| 断点 | 在案例中的表现 | 哪些信息能帮助区分 | 哪些仍属于 agent 的能力 |
| --- | --- | --- | --- |
| 目标/合法域不清 | 快的局部 fixture，不一定覆盖通用 request/mapping/logits | 输入约束、validation coverage、计时目标 | 判断候选是否满足真实需求 |
| 源码抽象与执行组织不一致 | `parallel` 仍产生多次 submit；一条大 SPMD task 分散很久 | 编排、task/grid/block 和完成边界 | 设计新的任务映射 |
| 缺少实际后端成本线索 | scatter 标量化；请求流水未兑现 | 精确 lowering、实际缓冲、pinned primitive | 发现等价表达和合法变换 |
| 指标与动作之间没有解释链 | scalar 高可能来自扫描、重排、同步或地址计算 | loop/work model、代码位置、局部 PMU 和反例 | 提出能区分解释的最小实验 |
| 搜索与结果记忆不足的风险 | 更多 blocks 未见收益，联合改动也不能单项归因；这些结果值得保留，避免后续重复或误用 | parent/diff、预期观测、不变量、失败结果、适用条件 | 选择探索顺序与预算 |

更好的反馈能降低“找到证据和建立联系”的成本，但不能保证 agent 发明 FlashAttention 一类算法，也不能修复缺乏合法变换能力的问题。必须分别评测**信息可达、解释正确、候选有效、实验接受**，不能把四者合并为“有 profiling / 没 profiling”。

## 4. 算子开发 agent 真正需要回答的问题

信息的价值应当以“是否改变了下一步选择，或排除了一个合理解释”衡量，而非字段数量或查询次数。

| 决策问题 | 最小有用信息 | 当前 pfdb 状态 |
| --- | --- | --- |
| 这两个结果真的可比吗？ | workload/输入语义/验证/metric/平台与工具链/采样契约 | 一部分已有；workload 与数值契约缺少结构化绑定 |
| 哪段计算影响最终完成？ | semantic stage、Observed contribution、边界、与其他工作的重叠 | task CP 已有；跨实现的阶段对应与汇总不足 |
| 慢在单 block，还是提交/分散启动/组完成边界？ | task span、row 分布、launch spread、直接依赖、grid/work mapping | rows 和部分 timing 已有；默认摘要与工作映射不足 |
| 为什么这些 pipe 忙或不忙？ | 同一计算的 PMU、实际操作与循环、layout、同步、buffer 结构 | PMU 已有；与代码/IR/实际实现的连接不足 |
| 是有用工作多，还是浪费多？ | 候选数、有效位置、padding、重算、重复读取及推导条件 | 不能仅从当前 timing rows 得到；需要源/IR或外部注解 |
| 这次改动验证了哪个解释？ | parent、确切 diff、不变量、预测、validation、完整 latency | trial 有基础；实验语义与有效性结构不足 |
| 当前证据还不能回答什么？ | 缺失原因、需要的工件类型、当前可查询范围 | modality 状态已有；与具体问题的关联不足 |
| 已有经验能安全迁移吗？ | primitive/axis/dtype/layout/shape/pin/输入域/反例 | 可用 notes/evidence refs 表达；没有可验证的条件化记录 |

一个完整的证据链通常是：

```mermaid
flowchart LR
    A[问题与合法输入域] --> B[实现与编译产物]
    B --> C[阶段 / task / block 对应]
    C --> D[时间与 PMU 观测]
    D --> E[候选解释与差分实验]
    E --> F[正确性与完整延迟结果]
    F --> E
    B --> G[实际 lowering / 缓冲 / primitive]
    G --> E
```

pfdb 应使这些边可以被查询和追溯；不必把每个节点都建成一个新子系统。

## 5. 当前系统：保留什么，具体卡在哪里

### 5.1 已经成立、不能重复造的基础

当前系统已提供：

- runtime 官方 Observed/static CP、非重叠 contribution 与 stall；
- END→FIN→dispatch→receive→start 的分解及有符号 early-dispatch 时序；
- physical rows、直接依赖、参数 shape/stride、scheduler/host/scope 视图；
- PMU 加权归一化与原始 sample provenance；
- 可选模态 requested/discovered/parse 状态和缺失原因；
- artifact SHA、backend revision、精确函数 hash、analysis version、兼容 ID；
- 有界 fact、显式 truncation、只读默认、render cache；
- baseline/trial、raw benchmark strata 和确定性 bootstrap。

这些应继续复用，不应因为新研究而另写 critical path、调度推断或独立 PMU 算法。代码依据：[关键路径与 PMU handler](../../profile_db/src/profile_db/query/handlers_z4.py#L415-L584)、[runtime adapter](../../profile_db/src/profile_db/runtime_adapter/__init__.py#L54-L153)、[运行时集成约定](pfdb-runtime-integration.md)。

### 5.2 对实际工作集做只读回放

以下为现有 compressor 数据库的实际结果。字节数指本环境 UTF-8 fact 响应，不是 token 数，也不是推测的 agent 成功率。

| 查询 | 完整事实数 | 默认输出 | 观察 |
| --- | ---: | ---: | --- |
| `overview`，run 1 | 4 | 585 B | 已经紧凑、完整 |
| `critical_path`，run 1 | 3 | 764 B | 能直接定位三段路径 |
| `critical_path`，run 7 | 47 | 13,126 B | 32 KiB 默认预算内完整；但长路径缺少面向阶段的首层摘要 |
| `tasks --family prefill_c4_softmax_pool`，run 7 | 16 | 6,304 B | 完整；每实例列举不等于得到跨 chunk 的完成顺序解释 |
| `rows`，run 2 projection | 64 | 4,041 B | 显示 25 行，剩余 39 行截断 |
| `perf_hints`，run 2 | 26 | 3,720 B | 显示 5 条，剩余 21 条截断；完整响应 15,837 B |
| `bench`，run 1 | 101 | 4,076 B | summary 后默认带所有 round，只显示 51 条事实 |
| `why_long`，run 6 第二个 projection | 1 | 231 B | `busy_us=1086.8`，同时 `max_row_us=109.08`，但没有分散启动等摘要 |
| `pmu`，run 1 fused tile | 2 | 232 B | 显示 unavailable；另一个目录中的独立 PMU-only capture 实际存在 |

注册表为 **27 个 query / 41 个 MCP tools，其中 8 个是写操作**。完整 tool schema 紧凑 JSON 为 **25,110 B**。实际客户端是否把所有 schema 都放进 prompt，取决于发现/延迟加载机制；不能据这个字节数宣称 agent 一定被工具目录压垮。比数量更有力的证据，是下面的可达性问题。[工具生成实现](../../profile_db/src/profile_db/mcp_server.py#L230-L272)

### 5.3 已复现的接口缺陷：有数据，却不能顺利拿到

**A. 截断后的 MCP 续取路径缺失。** `rows` 等响应给出 `retry --budget ...`，但对应 MCP schema 没有 budget/cursor，dispatch 固定使用 query 默认预算。`rows`、`perf_hints` 也没有独立分页 selector。CLI 可以加预算，因此底层数据不是缺失；对仅使用 MCP 的 agent，这条恢复指令不可执行。[截断提示](../../profile_db/src/profile_db/facts.py#L96-L149)、[MCP dispatch](../../profile_db/src/profile_db/mcp_server.py#L380-L387)、[RowsParams](../../profile_db/src/profile_db/query/params.py#L183-L184)

**B. `tasks` 的 name 筛选碰撞。** API 的 `query(self, name, **params)` 与 query 参数 `name` 重名。实际调用：

```python
db.query("tasks", run_id=7, name="prefill_c4_softmax_pool")
# TypeError: ProfileDB.query() got multiple values for argument 'name'
```

MCP 同样转发关键字参数；CLI 的 `--name` 路径也经过同一个 API，因此受影响。registry 直接接收参数 dict 时能返回 16 条事实，exact family 筛选也可用。该 TypeError 还会绕过只处理 `PfdbError` 的公开入口。这是传输/接口缺陷，不是 profiling 能力缺失。[API](../../profile_db/src/profile_db/api.py#L106-L112)、[参数](../../profile_db/src/profile_db/query/params.py#L135-L141)、[CLI](../../profile_db/src/profile_db/cli.py#L482-L515)、[MCP](../../profile_db/src/profile_db/mcp_server.py#L385-L415)

本任务按“只研究”的要求没有修复以上问题。

### 5.4 有指标，但它没有表达当前要判断的事情

**时间口径。** `task.busy_us` 当前计算为 `max(row.end)-min(row.start)`，而 `task.wall_us` 为 `max(FIN)-min(dispatch)`。前者不是累计每核 active time。`why_long` 已提供 row min/max，不能说“完全没有 block 信息”；但第二 chunk 的 1,086.8 µs 与最长单 row 的 109.08 µs 之间，需要 launch spread、row busy sum、并行波次或等待关系进一步解释。[实际聚合](../../profile_db/src/profile_db/ingest/__init__.py#L111-L138)、[why_long](../../profile_db/src/profile_db/query/handlers_z4.py#L193-L238)

**比较口径。** 当前 compatibility gate 核对 program、level、clock、core topology、分析版本/函数等。它没有核对 workload shape、输入快照、position/mapping 分布、validation 和数值契约。实际回放中，已知 T=512 的 run 1 与 T=8,192 的 run 6 得到 `compatible=true, profile_compatible=true`。这只证明当前 gate 所核对的条件成立，不能解释为同输入的优化比较。[compare](../../profile_db/src/profile_db/lifecycle/compare.py#L32-L85)

**优化指标。** median 已经存储并在 overview/bench 显示；但 compare、默认 baseline、bootstrap 和部分列表以 mean 为中心。mean 计算并未因此错误，问题是当前[测量规则](../../.claude/rules/benchmarking.md#L24-L27)偏好 median，而系统没有显式的 campaign objective 来阻止调用方在两者间换口径。[比较指标](../../profile_db/src/profile_db/lifecycle/compare.py#L24-L29)、[baseline](../../profile_db/src/profile_db/lifecycle/baseline.py#L35-L58)、[bootstrap](../../profile_db/src/profile_db/lifecycle/bootstrap.py#L50-L75)

**跨实现阶段。** family delta 按字符串 family 对齐，并对 task 的 busy/wall 求和。SWA 前后结构、compressor fused/split 结构不能靠名字自动表达同一语义阶段；长序列中重叠 task span 求和也不等于阶段 elapsed。当前 delta 不应被当作程序总延迟的可加预算。[family 聚合](../../profile_db/src/profile_db/lifecycle/compare.py#L89-L143)

### 5.5 数据存在于工件层，但没有安全的连接

本次 compressor 数据库有 7 个 run、35 个 artifact，而 `pmu_counter=0`、`memory_entry=0`、`trial=0`。这些数字不是“研究没有用 PMU/实验”的证据：PMU 存在于独立目录，历史研究在 pfdb 外聚合；实验记录主要在研究脚本、日志、报告和 CSV。

当前初次 ingest 会读取 capture 目录中的 `pmu.csv`，但 generic attach 的 optional discovery 只覆盖 args/scope/incore/host 等，没有 PMU。因此“PMU 查询能力已实现”与“独立 PMU-only capture 可以方便绑定到本次 timing”是两回事。[初始 ingest](../../profile_db/src/profile_db/ingest/__init__.py#L226-L250)、[attach discovery](../../profile_db/src/profile_db/ingest/analysis.py#L27-L49)

memory report 缺失则不能直接归咎于 pfdb parser：此前编译没有生成该报告。实际 `.pto` 仍包含布局与地址信息，但系统尚未把它与 task/kernel 资源证据关联。这需要区分**没有采集、没有导入、没有绑定、没有派生、只在展示上难用**五类问题。

## 6. 哪些信息应该收起，哪些不能删

“冗余”是相对于当前问题的，不是某类工件天然没有价值。建议三层访问：**首层判断事实 → 定位后的局部证据 → 可复查的完整工件**。

| 信息 | 默认处理建议 | 仍需保留的原因 |
| --- | --- | --- |
| benchmark 每个 round | 默认 summary、分位数、strata/配置；samples 按需 | 判断噪声、漂移、离群和置信区间 |
| 全部 PMU counters / 所有 samples | 默认目标阶段的少量有定义的 pipe 指标；完整数据可取 | 区分模型解释、核间差异、事件组和缺失 denominator |
| 全部 physical rows | 默认分布、启动/结束展开度、极值 row 的句柄 | 尾波、early dispatch、core blocker 证明必须钻取 |
| 全部 compiler hints | 默认与目标 kernel/源码区域相关的提示 | 少数看似低频的 buffer fallback hint 很关键 |
| 所有 scheduler/host/scope 相位 | 对已选窗口、task 或问题请求 | 调度和 host 成为问题时不可替代；不能因这两例不常用就删 |
| 每行重复 run/family/单位定义 | 放入响应公共上下文和 metric 定义，保留局部稳定 ID | 可核查，且避免表格丢上下文后无法解释 |
| 全量 hash / 函数源 fingerprint | 默认短 handle 与兼容状态；细节按需 | 安全关联、升级复算和审计需要 |
| 全量泳道图或 Perfetto JSON | 默认不给，窗口图/导出按需 | 图有助于发现长序列空窗；不需要把整个 JSON 塞进上下文 |
| 未实现模态的长串 unavailable | 首层压缩为问题相关的缺失摘要 | requested/absent/parse_error 不能混为“没有问题” |

已有的 bytes budget 是必要条件，但**先输出不相关内容，再截断真正需要的内容，不构成有效的信息压缩**。按问题筛选、摘要先行、可恢复分页，比统一缩小 budget 更重要。

也不能只返回几个“最大百分比”的 counters：没有 stage、分母、采样范围和代码联系，80% scalar 与 50% MTE2 都可能被错误解读为确定瓶颈。压缩必须保留判别力。

## 7. 最值得补齐的六类连接

### 7.1 比较条件与数值验证：让“可比”有明确含义

建议在现有 run/artifact/trial 之上补充小型、版本化的外部上下文记录，覆盖：

- **实现身份**：源与实际编译产物 digest、入口/ABI、编译配置、工具链版本。
- **工作负载身份**：逻辑 shape/dtype/layout、请求/序列分布、相关 scalar、输入/fixture 快照身份。
- **合法域与验证**：参考实现、comparator/tolerance、测试集/输入分布、pass/fail/unknown、已覆盖/未覆盖条件。
- **目标与测量范围**：metric=median 或 mean、计时窗口、warmup/rounds/strata、rank aggregation、单算子或模型上下文。

不是要求所有比较都使用完全相同的文件名或原始 binary hash。布局变换、等价输入表示可有显式逻辑身份映射；但“相同 shape”“相同随机种子”不能自行升级为语义等价证明。

比较应区分三种用途：

1. **描述性比较**：允许看不同 shape 的 scaling，明确哪些条件变化，不输出同 workload 的 win 结论。
2. **优化效果比较**：要求目标、workload、验证与测量契约兼容；未知条件显示 unknown，不能默认通过。
3. **机制证据比较**：可以结合分开采集的 CP/PMU/编译信息，但必须显示其来源，不能假装同一次时间轴。

pfdb 不替 agent 跑 golden，也不证明全部数值正确；它验证外部证书与当前 implementation/workload 的关联，并展示证书的边界。caller 声明的“position 连续”与工件确证的连续性，应保留不同来源。

### 7.2 从 family 名称转向可追溯的计算阶段

这里的“计算阶段”指待比较的数学工作，例如 attention/RoPE、A projection/quant、B projection；不是要求框架新增一种 runtime task。

一个阶段可能对应多个 task，也可能只占 fused task 的一部分。建议支持显式映射及其来源：

- compiler/source location 可证明的映射；
- 外部 harness/开发 agent 提供、尚未独立验证的标注；
- 无法分离的 fused region，标记为 indivisible，不能臆造子阶段耗时。

跨版本映射必须支持一对多、多对一和 unresolved。名字相同但工作变化时也不能直接认定对应；结构变了时，不应把不存在的 family 记为“性能归零”。

阶段摘要应区分 Observed contribution、task span、row 生命周期累计 core-time、窗口并集；生命周期时间包含同步与等待，不等于有效算术或 pipe-active 时间。融合阶段不可分时，只提供可测的外包络和相关代码。**不同 stage 的独立时间包络也可能重叠，不能承诺它们逐项相加等于总时间。**

这能将 SWA 的“完整前段替换”与投影不变性组织成清楚的对照，也能避免 compressor 中把更快投影误读为整算子更快。

### 7.3 从 task span 到 block 工作与完成边界

建议复用现有 rows，先补小型摘要，而不是新增一套 scheduler：

- row duration 的 min/median/P90/max；
- earliest/latest start、launch spread、tail；
- `sum(row.end-row.start)`，明确为 kernel 生命周期累计 core-µs，包含同步与等待；
- 已知 grid、实际活跃 rows、compatible cores；
- 输出/组/chunk 与 task/block 的对应，缺失则显式 unknown；
- 依赖的完成边界：所有组、本组、tensor region、显式控制边。

需要区分三种不同粒度：**提交粒度、core 上单 block 工作粒度、producer 对消费者释放完成的粒度**。SWA 分别证明了合并 submit 与缩窄组依赖的价值；它们不等于增加 block 数。

工作量信息也分证据等级：

- 由源码/IR 证明的循环次数及重算范围，可以记录为结构事实；
- 依赖实际数据的 useful work，要由输入工件、计数或外部声明补充；
- 不能从 row 持续时间自动猜测“有效输出数”“HBM 字节数”或“浪费 25% 时间”。

例如 compressor 后续 chunk 有 25% 冗余 projection FLOPs，是有前提的工作量推导；它不意味着总 latency 会减少 25%。

### 7.4 连接到实际 lowering，而不是只给硬件规格表

优先目标是有限的、可复查的索引：

```text
task / kernel handle
  → DSL source span
  → generated function 与 .pto/.cpp span
  → tile/layout / loop / buffer / primitive form
  → 对应工具链 pin 的实现与已知约束
```

两个案例最有价值的条目是：

| 静态证据 | 应输出什么 | 不应输出什么 |
| --- | --- | --- |
| 特定 TSCATTER 路径包含逐元素标量循环 | axis/mask/dtype/layout、文件/行、pin、静态调用点，说明动态次数是否已知 | “所有 scatter 是瓶颈”“换 gather 必快” |
| stage=2 发生 buffer alias | requested depth、realized buffers、相关 hint、空间/限额与别名 | 没有隔离实验就声称损失多少 µs |
| M/N/K 与 L0 tile 改变 | 源 tile 与 realized tile、双 accumulator、X 复用、缓冲高水位 | 仅按容量百分比打分 |
| pool 扫描完整 tile | 候选循环边界、数据依赖条件、useful candidate 的证明/声明 | 把 metadata search 当作等量向量计算 |

当前 hints、memory、incore 已覆盖其中一部分。优先接现有编译产物与 runtime/编译器结构化工具；只有上游没有可复用输出时，才考虑小范围离线索引。无法从现有 artifact 确认的内容返回 unavailable，向采集方描述所需 sidecar，不能修改 runtime 或伪造数据。

不建议一开始建设通用 IR 解释器、指令性能模拟器、自动 roofline 推断或新的硬件知识图谱。静态调用点数、动态执行次数、pipe busy、带宽、理论峰值不是同一个量。MTE2 busy 也不能单独证明达到了 HBM 峰值。

### 7.5 安全关联独立采集：尤其是 PMU-only

这是比增加 counters 更直接的需求。允许关联已有 sibling capture，但至少核对：实现/编译产物、入口、workload/相关控制值、平台与 profiling 配置，并保存调用方声明与不确定项。

不同 capture 的 task ID、func ID 或文件名相同，不足以证明相同工作；`run_epoch` 等仅在相应 capture 的身份域内有意义。task/block 对应无法确定时，最多做带范围声明的 kernel/family 聚合；不要做逐 row join。

同时保留：

- 哪个 capture 给时间，哪个给 PMU，哪个给 compiler evidence；
- PMU event group、sample coverage、denominator 与加权方式；
- 可比较与不可比较的指标；
- 合并匹配依据和 unresolved 映射。

这样 agent 才能在一个目标对象上取到相关证据，而不是每次离开 pfdb 写一次 CSV join。该能力不要求 pfdb 自己发起采集。

### 7.6 结构化实验，而非仅有一段 hypothesis 文本

现有 trial 已支持 parent、hypothesis、changed_files、verdict、evidence refs；应在此基础上增强，不重新做一个搜索平台。[trial 实现](../../profile_db/src/profile_db/lifecycle/trial.py#L26-L79)

一份最小实验记录需要回答：

| 字段/关系 | 作用 | 案例 |
| --- | --- | --- |
| parent 与不可变 change-set 身份 | 防止比较错 baseline 或仅记文件名 | B 依赖只改 all-groups→own-group |
| hypothesis 与预测的中间观测 | 明确该实验试图区分什么 | gather 后 scalar cycles/融合阶段时间预计下降 |
| 保持不变的条件及其证据 | 限制因果解释 | tile、block、kernel body、输入、comparator 保持 |
| 结果类型与 validation | 将 compile/runtime/precision failure 和有效回退分开 | 组合 projection 结果不正确，不是性能回退样本 |
| measured outcome 与 caller verdict 分离 | 保持数据库中立 | delta 是事实；“接受该方案”由 agent 决定 |
| applicability / counterexamples | 避免错误迁移 | scan-8 仅连续 position；128 blocks 未见收益 |

`changed_files` 相同不代表只改了一个因素；没有 diff 或不变量证据时，显示 uncontrolled，不自动称为 causal。当前 `set_verdict("win")` 不要求 correctness evidence，这可以保留为 caller 标签，但不能让其自动等于“已验证有效优化”。

trial 行当前在 prune 后保留，而完成 trial 的 run 不一定受保护；现有保护主要是 baseline、active trial 和最新若干 run。建议区分**可清理的大工件**与**值得保留的小型条件化实验摘要**。完成实验至少导出/保留结果、条件、hash 与证据定位，避免只剩一个失效 run ID；不必无限保留所有 captures。[工作集规则](../../profile_db/src/profile_db/lifecycle/working_set.py#L21-L42)、[prune](../../profile_db/src/profile_db/lifecycle/prune.py#L28-L92)

当前 compressor 数据库 trial 为 0，说明这次实践没有使用它来承载实验记忆；不能据此断言所有调用方都会不用。需要观察真正的使用轨迹再判断录入成本和收益。

## 8. 面向 agent 的接口应该如何收敛

### 8.1 先收敛用户问题，不先冻结新 API

建议以三个读工作流组织默认入口；下面的名字是说明用候选，不是实现承诺：

| 工作流 | 核心问题 | 返回内容 |
| --- | --- | --- |
| `brief` | 我在优化什么？哪些范围值得继续看？ | 契约/有效性摘要、完整 latency、CP 阶段与关键缺失 |
| `inspect` | 这个 stage/task/window 的证据是什么？ | 相关 timing、block、依赖、PMU、lowering，附稳定证据句柄 |
| `compare` | 这次改动改变了什么，结果是否可接受比较？ | 条件差异、semantic mapping、各测量 delta、实验有效性与未决项 |

写入/attach/refresh/实验记忆是可选的独立工作流；低层 CLI、现有 handler 和完整证据仍可保留。只读 agent 不必先学习所有维护操作。

最小实现探索应先尝试**复用现有查询组成一个 brief 和局部 inspect**，而非同时重写存储、schema、runtime adapter 和所有入口。若聚合成一个巨大带几十个条件参数的工具反而更难调用，应保留更清楚的细分。减少工具数本身不是验收标准。

### 8.2 一个局部响应应当包含的内容

以 compressor pool 为例，理想回答的逻辑顺序是：

1. **对象与范围**：哪个实现、workload、stage；验证覆盖什么。
2. **与目标有关的时间**：完整 benchmark、stage 的 CP contribution 与 AICore span，明确不同捕获不能混减。
3. **能区分解释的结构**：128 blocks、40 AIV、每输出扫描 512 候选；后两项的来源和推导前提。
4. **相关硬件证据**：scalar busy、vector busy、MTE2，附采样分母与 capture 关系。
5. **源码与已实现行为**：循环位置、realized code、可取的详细证据句柄。
6. **尚未成立的部分**：通用 mapping 下是否能收窄、是否只有当前 fixture 通过、是否完成隔离实验。

它不应直接输出“瓶颈是 scalar，所以改成 scan-8”。前五项使 agent 能提出这个假设，第六项防止它把局部捷径直接写进通用算子。

### 8.3 信息预算与证据标签

建议默认摘要、显式 detail、稳定分页；完整 rows/hints/path 必须能通过当前通道续取。输出排序应优先契约与问题相关证据，而非数据库插入顺序。大量重复定义放公共上下文；稳定 task/row/artifact handle 保留，避免为了短而丢失可钻取性。

沿用现有 measured/proven/unproven/unavailable，补充必要的来源与适用范围，而非另建一套 confidence 分数：

- `origin`：capture / runtime derivation / compiler / caller declaration / experiment；
- `scope`：具体 workload、阶段、capture 和版本；
- `validity`：validation 的 pass/fail/unknown 与覆盖域；
- `missing`：要回答本问题还缺什么。

“caller 声明的假设被测量系统收到了”是一个事实，但假设内容不因此变为 proven。“critical path membership”也是观测事实，不能直接推出把它优化 10% 就能让整算子快某个固定比例。

## 9. 职责边界与复用策略

| 责任方 | 应承担 | 不应交给它 |
| --- | --- | --- |
| runtime / compiler | 官方计时/图/调度/导出及其定义 | 本研究不修改这些仓库 |
| 外部采集与 golden harness | 编译、设备/模拟器执行、输入与 validation、采集 manifest | 不把采集偷偷塞进 pfdb query |
| pfdb | 读取现有工件、校验关联、复用官方分析、形成有界证据包和实验记录 | 自动执行优化、自动采集、无证据地宣判瓶颈 |
| 算子开发 agent | 解释事实、提出变换、设计对照、执行采集、决定接受/拒绝 | 让它重复解析海量 JSON 或猜测指标口径 |
| 仓库中的条件化经验/文档 | 保存 primitive/布局/pin 约束和已测反例 | 把经验写成不带条件的永久优化定律 |

对于 runtime 已覆盖的能力，复用其结构化入口和现有 adapter；不同口径仍以 runtime 为准。runtime 的缺陷只在 pfdb 一侧按已知版本做窄兼容，并继续维护[集成台账](pfdb-runtime-integration.md)。

新增证据类别也需要记录 producer/version、输入 artifact hash、提取方式和退役条件。如果未来上游提供可靠的 task→source 或 block→work mapping，pfdb 应替换本地索引，避免同时维护两份语义。

外部经验不必塞进 DuckDB 成为新长期知识库。可以先使用已有 trial/evidence refs 与小型导出文档；保存“该 pin 上该 form 的 gather 对照获益”，而不是自动训练“scatter 应被替换”。

## 10. 外部第一方研究的支持与限制

外部研究只能帮助提出系统假设。以下 CUDA、GPU、CPU 结果都不能当作 Ascend/PyPTO 的性能预测。

| 来源与版本 | 与本问题直接相关的发现 | 对 pfdb 的启示与限制 |
| --- | --- | --- |
| [KernelBench v1，2025-02-14，§5.1/Table 1](https://arxiv.org/html/2502.10517v1#S5.SS1) | DeepSeek-R1 L2 fast₁ 单次基线为 36%；在 10 次调用预算下，历史生成 G＋执行反馈 E 为 62%，再加 profiling P 为 72%；另一些模型/层级加入 P 反而下降 | 执行、正确性、profiling 信号应分开；反馈价值取决于模型和任务。不能由“接入 profiler”推断必然提高优化能力 |
| [KernelPro v2，2026-08-08，§5.3/Appendix C.1](https://arxiv.org/html/2606.26453v2#S5.SS3) | 42 个选定任务，Sonnet 4.6/A100、相同 greedy search；无 profiling / 约 50 个 raw NCU metrics / 结构化工具的几何平均 speedup 为 3.35× / 1.77× / 4.00× | 作者报告的 preprint；每任务取 15 seeds 最好结果，30 iterations，失败记 1×，有 outlier cap。支持测试反馈组织方式，不证明 pfdb 应复制其 heuristic 诊断或其收益 |
| [FastKernels v2，2026-10-01，§6.3](https://arxiv.org/html/2605.23215v2#S6.SS3) | 作者报告 isolated kernel 的 1.6–6.6× 优势在端到端至多 1.25×；只有 20% 的 winning kernel sets 可直接正确运行 | 应绑定输入/接口/数值/部署范围。20% 指 kernel sets，不是本项目单算子失败率 |
| [Anthropic 工具设计实践，2025-09-11](https://www.anthropic.com/research/writing-tools-for-agents) | 建议围绕高影响工作流组织工具、返回相关上下文、提供 concise/detail，并用实际 agent 评估调用与结果 | 第一方工程经验，不是 pfdb 对照实验；没有给出通用最优工具数或字节预算 |
| [Coz，SOSP 2015；arXiv 版本 2016](https://arxiv.org/html/1608.03676) | 时间占比与 critical-path membership 仍不等于优化收益；其因果证据来自运行时干预 | 离线 pfdb 应区分观测、模型和已验证改动；不能从一份 trace 自动给因果 speedup。Coz 的 CPU 干预机制不是本项目实现方案 |
| [PyTorch KernelAgent，2026-03-06](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) | 架构将采集、分析、建议、候选探索和正确性/性能测量分开 | 支持职责分离作为参考；不是必须采用同样多 agent 架构的证明 |

对本研究最重要的共同点是：**信息访问、解释、动作生成和验证是不同环节；增加 raw metrics 甚至可能降低优化质量，必须在实际 agent 上做消融。**

## 11. 如何验证改进确实有用

当前测试验证了数据解析、确定性、口径、truncation 和脚本式导航，是必要基础。但预定义的 list→overview→density→task 路线成功，并不能证明 agent 更会选优化方向。[golden QA](../../profile_db/tests/golden_qa/test_golden_qa.py#L44-L87)、[MCP session 测试](../../profile_db/tests/test_mcp_server.py#L151-L165)

建议分两级评测，先证明信息设计有价值，再决定是否扩展实现。

### 11.1 离线决策题：只读已有工件

用两次实践重建题目，不把研究报告答案直接给被测 agent：

| 题目 | 合格答案应包含 | 常见错误 |
| --- | --- | --- |
| SWA parallel 是否已合并提交？ | submit、grid、block、物理行的区别，引用编排证据 | 看 DSL 关键字回答“已经并行” |
| scalar 高可以证明 scatter 瓶颈吗？ | 多个解释；索要局部 lowering，引用版本与 form | 直接宣布所有 scatter 都慢 |
| B 投影为何等其他组？ | 直接前驱/region、组内独立性、窄完成边界 | 建议删除所有依赖 |
| stage=2 是否兑现？ | 实际 buffer 数/别名与相关 hint | 只引用源码 stage 参数 |
| compressor 较长 projection 是单核计算慢吗？ | task span 与 row duration/launch spread 的区分 | 将 1 ms 写成单 matmul 时间 |
| 512 与 8,192 tokens 能作为同 workload 优化比较吗？ | 拒绝该解释；允许描述 scaling | 因 program/topology 相同就报告 speedup |
| scan-8 能直接替代通用版本吗？ | 说明 position/mapping/请求约束及覆盖缺口 | 以当前 PASS 当作全面正确性证明 |
| 更高 occupancy 是否更好？ | 用相同目标的 wall time 判断，识别无用工作 | 奖励手写 pool 更高的占用率 |
| precision fail 的候选是否能排名？ | 单独记录无效结果，不按 latency 接受 | 放宽容差、把失败当 0 µs 或简单回退 |
| MCP 截断后如何取得完整关键证据？ | 在允许的接口内可续取，或明确接口不可达 | 忽略剩余项；幻觉“全部看过了” |

这一级衡量：事实正确率、错误因果归因率、证据引用可复查率、正确拒绝无效比较的比例、工具错误与恢复、首次形成可检验假设所需的调用/字节/token/时间。开放假设题可有多个正确方向，不能只奖励复述既有研究的唯一方案。

这些题目用于开发验证；不能再拿同一批题宣称泛化。保留其他 shape/operator 或未见反例作 holdout，也要防止报告正文和公开解答泄漏到测试上下文。

### 11.2 端到端优化评测：由外部执行器负责

建议至少对照：

- A：当前 pfdb 与同样的原始工件访问权限；
- B：A + 有界关联摘要/可执行钻取；
- C：B + 代码/lowering/work/验证关联；
- D：C + 结构化实验记忆。

保持 agent/model、DSL 文档、候选起点、输入、工具链、硬件条件、预算和可访问的手写参考一致。一次只改变一类反馈，避免把更强模型、更多设备时间、更丰富原始数据或已知答案的提供混作 pfdb 收益。固定模型版本并多 seed；参考可见与不可见应作为不同实验条件。

保留可观察的工具调用与响应、代码改动、公开的假设/决策记录和验证结果即可；不依赖或要求获取模型的隐藏推理过程。这些轨迹用于区分“信息未拿到”“信息被误解”“没有生成有效候选”和“实验没有通过”。

主指标是**预算内最终、独立验证通过的完整算子 latency 与成功率**；同时记录无效候选率、回退率、首次有效收益时间、重复假设、调用/上下文成本、验证覆盖、不同输入上的稳健性。失败样本不能从总体统计中悄悄删掉，profiling 的 observer 时间不能用作优化目标。

对 winner 再用一致的独立 benchmark strata/配置和选定 estimator 确认；平均值的 CI 不能冒充中位数的 CI。选中候选后的最终测量与搜索中的选择偏差也要分开处理。硬件采集始终留在 harness/执行器中，不进入 pfdb。

还应逐项移除 CP、PMU、lowering、实验记忆等信息做信息消融，观察哪种信息对哪类问题有益。若更简洁的摘要降低错误调用，却没有改善正确性或完整 latency，应如实记录；不据工具调用次数更少直接宣布系统成功。

以上评测本次均未执行。

## 12. 优先级与暂不建议做的工作

| 优先级 | 建议方向 | 为什么现在值得做 | 在本研究中的状态 |
| --- | --- | --- | --- |
| P0 | 修复公开入口 name 碰撞、MCP 截断续取与错误路径 | 已有证据无法可靠访问；有只读复现 | 已确认问题，未改实现 |
| P0 | 明确 objective、比较用途与验证/workload 绑定 | 已复现不同 shape 被标 compatible；防止接受无效胜利 | 缺口已确认；字段/接口需评审 |
| P1 | 基于现有 rows/CP 的阶段与 block 摘要 | row/CP 统计可由现有数据派生；阶段映射需显式来源 | 有案例依据，收益待评测 |
| P1 | 独立 PMU capture 的安全关联与局部联合视图 | compressor 的独立 PMU-only capture 需要在 pfdb 外关联 | 工作流缺口已确认 |
| P1 | task→source/generated primitive 与 realized buffer 索引 | 两例的关键解释都跨过这一层 | 限定范围与上游复用方式后验证 |
| P2 | typed 实验结果、不变量、失败与适用域、小摘要保留 | 支持正确归因和有条件迁移 | 现有 trial 可复用，勿先重建搜索系统 |
| P2 | 用实际 agent 验证默认入口、信息筛选和工具发现 | 工具数与信息量本身不能证明优化收益 | 设计了实验，尚未执行 |

暂不建议：再造 runtime 关键路径工具、把 runtime CLI 全量包装成 MCP、先做更多 counters 的全量导入、通用指令模拟器、自动“瓶颈→建议”规则库、无条件经验 RAG、无限保存全部 traces，或为了压缩工具数改成一个难用的万能接口。

需在后续设计中明确但不阻碍本研究结论的问题是：谁提供可信的 workload/validation manifest；阶段与 work mapping 哪些由编译产物确证、哪些必须外部标注；实际部署的 agent 主要通过 MCP 还是 CLI；愿意用多少额外上下文/离线分析成本换取多少优化收益。这些应由小规模原型与评测回答，而不是先扩张数据库。

**最终研究判断：pfdb 的价值应从“收集并提供了多少 profiling 信息”转向“能否让 agent 以更少的无效推断，形成并验证正确的性能假设”。当前基础值得保留，首先应修通已有证据的访问与关联，再以真实 agent 的结果验证新增信息是否有用。**

## 13. 本次交付与未完成验证

本次仅新增本文。保留全部先前研究文件和本地工件；pfdb、算子、runtime 与数据库均未修改，未发起新的采集或上游 issue。

完成了案例证据复核、当前源码/测试审查、公开入口与响应预算的只读回放、数据库 hash 不变检查和第一方文献核对。尚未进行新接口实现、agent A/B、跨模型/多平台泛化或新的数值覆盖测试；所有系统收益建议均保持为待验证假设。
