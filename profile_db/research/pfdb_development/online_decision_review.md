# 在线优化试验的公开决策与结果复核

本复核发现了三类可区分的问题：**指标含义已经给出，提案仍将生命周期当作活动时间；编译器资源提示已经给出，提案仍混淆内存池或把消除提示当作性能目标；变换改变了任务、缓冲或跨核交接，却没有相应的合法性或生成物证据。** 同时出现了有希望的搜索候选，不能将本轮描述成 agent 普遍失败。

**2026-10-21 的本次复核覆盖全部 24 个计划会话、44 个公开尝试结果：30 个 `search_validated`、10 个 `failed`、4 个 `agent_stopped`。** 失败中 3 个来自模型进程，7 个来自候选执行进程；不能合并解释成数值失败。[试验汇总](../../../build_output/pfdb_development/optimization-pilot-v1/results.json) 记录 `original_inputs_unchanged=true`。搜索已完成；本报告未读取正在进行的独立确认，所有搜索会话仍将 held-out 验证和独立性能确认标为 `pending`。

## 证据与判断边界

- 只读取完整会话的结构化公开提案、源码差异、结果、验证日志、编译提示及已有生成物。未重新编译、运行模型或使用 NPU；未修改输入、capture、数据库或上游代码。
- 试验配置见 [protocol.json](../../../build_output/pfdb_development/optimization-pilot-v1/protocol.json)：每会话最多两个候选，非 profiling 搜索计时为 20 次、预热 5 次；模型预算 180 秒；串行执行。replicate 编号不是受控随机种子。不能把此轮计时与两个历史案例报告中的不同轮次计时直接混合。
- `baseline` 与 `semantic_contract` 共用基础 pfdb 事实；后者额外提供指标定义和验证边界。这是一个组合条件，不能分离“指标定义”和“验证边界”的独立作用，更没有检验所有 pfdb 信息是否必要。
- **这是固定初始证据包、工具禁用的短预算试验，不是可自适应查询 pfdb 的完整优化工作流。** [执行器](run_optimization.py) 第 62 行禁止模型工具，第 431–433 行的反馈只包含前一尝试的公开记录和当前 incumbent 源码：提案、源码 SHA、范围检查、进程状态/耗时、搜索样本与结果。**不会回传新候选的 compiler hints、memory allocation、level-4 或 PMU。** 这些工件可供本复核读取，模型不能打开 capture。后文必须区别“初始事实被误读”和“新候选事实没有提供”，不能把后者说成 agent 忽略了可访问的新证据。
- `search_validated` 表示本次冻结 fixture 的现有比较器通过并得到搜索计时，不是完整输入域等价性证明。AST 范围检查通过也不是数值或同步合法性证明。
- 下述“提案认为”只引用公开 `proposal.hypothesis`、`invariants` 和 `edits`，不推断隐藏推理、模型知识或原开发 agent 的思考过程。计时变好也不自动证明提案声称的机制。

## 完整会话清单

下表的 C0/C1 是会话内候选序号；第二个候选从当时保留的 incumbent 出发，未必累积第一个被拒候选。数值均为本轮搜索中位数，单位 µs。SWA agent、SWA manual、C4 agent、C4 manual 的初始搜索中位数分别为 **278.9、195.7、211.95、249.35**。

| 起点 / 条件 / replicate | 公开改动与有界结果 | 复核结论 |
| --- | --- | --- |
| [SWA agent / baseline / 0](../../../build_output/pfdb_development/optimization-pilot-v1/swa_agent-0-baseline/result.json) | C0 模型到 180 秒预算，无可执行提案 | 无法评价内核变换或 pfdb 事实正确性 |
| [SWA agent / baseline / 1](../../../build_output/pfdb_development/optimization-pilot-v1/swa_agent-1-baseline/result.json) | 同上 | 归入执行预算结果，不能当成数值失败 |
| [SWA agent / semantic / 0](../../../build_output/pfdb_development/optimization-pilot-v1/swa_agent-0-semantic_contract/result.json) | C0 删除 `pl.mul(acc_out, 1.0)`，数值验证失败 | Right 与 Vec 混淆，且交接结构改变；见已有失败复核 |
| [SWA agent / semantic / 1](../../../build_output/pfdb_development/optimization-pilot-v1/swa_agent-1-semantic_contract/result.json) | C0 将 A 投影拆成 group×N 任务、分离量化，编译阶段失败 | 具体原因未知，不能赋予数值结论；见已有失败复核 |
| [SWA agent / baseline / 2](../../../build_output/pfdb_development/optimization-pilot-v1/swa_agent-2-baseline/result.json) | C0 A 投影 stage 2→1：323.45；C1 A 投影改 32-block SPMD、分离量化：283.75；均拒绝 | 关闭失效深度、增加并行暴露均不保证更快 |
| [SWA agent / semantic / 2](../../../build_output/pfdb_development/optimization-pilot-v1/swa_agent-2-semantic_contract/result.json) | C0 同样 stage 2→1：323.75，拒绝；C1 gather 与 attention 两处改 SPMD：249.7，暂选 | 组织形式改动有搜索收益；不能拆分两处各自贡献 |
| [SWA manual / baseline / 0](../../../build_output/pfdb_development/optimization-pilot-v1/swa_manual-0-baseline/result.json) | C0 A/B K 256→128：199.7，拒绝；C1 M 16→8：失败 | 小 K 并非无效变换，小 M 的物理合法性未建立 |
| [SWA manual / baseline / 1](../../../build_output/pfdb_development/optimization-pilot-v1/swa_manual-1-baseline/result.json) | C0 A/B K→128：200.55；C1 A 的 N 128→64：198.85；均拒绝 | 更小缓冲、更细粒度都未形成搜索收益 |
| [SWA manual / baseline / 2](../../../build_output/pfdb_development/optimization-pilot-v1/swa_manual-2-baseline/result.json) | C0 A/B K→128：200.2，拒绝；C1 停止 | “仍未实现流水”不是计时可单独证明的事实 |
| [SWA manual / semantic / 0](../../../build_output/pfdb_development/optimization-pilot-v1/swa_manual-0-semantic_contract/result.json) | C0 A 的 N→64：201.0；C1 复用 amax 前的 slab，删第二次源码 load：199.05；均拒绝 | 未证明实际少一次搬运，也未证明由量化 GM 开销主导 |
| [SWA manual / semantic / 1](../../../build_output/pfdb_development/optimization-pilot-v1/swa_manual-1-semantic_contract/result.json) | C0 A/B K→128：194.7，暂选；C1 再降 K→64：199.6，拒绝 | 仅 1.0 µs 搜索优势；相同源码在另三个会话反而较慢 |
| [SWA manual / semantic / 2](../../../build_output/pfdb_development/optimization-pilot-v1/swa_manual-2-semantic_contract/result.json) | C0 仅 A K→128：200.05，拒绝；C1 M→8：失败 | 不能将“半数逻辑行为 padding”直接换算成半数物理 MAC |
| [C4 agent / baseline / 0](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-0-baseline/result.json) | C0 M64/N128/20 lanes：245.9，拒绝；C1 M256/N256/16 lanes：失败 | 均衡计算量忽略 tile 代价；大 M 触及已有容量警告 |
| [C4 agent / semantic / 0](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-0-semantic_contract/result.json) | C0 M112/N256/20 lanes：198.85，暂选；C1 停止 | 搜索收益与“更均衡”相容；停止提案的 unit 数计算错误 |
| [C4 agent / baseline / 1](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-1-baseline/result.json) | C0 仅 lanes 32→20：206.55，暂选；C1 RMS 从 16 行改 1 行并删 GM staging：失败 | C0 将实际 32 units 写成 128；C1 尚无可用 capture 或数值结论 |
| [C4 agent / semantic / 1](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-1-semantic_contract/result.json) | C0 仅 lanes 32→20：206.1，暂选；C1 删归一化后的 GM staging 并提升模板 load：失败 | C0 将实际 32 units 写成 256；C1 尚无可用 capture 或数值结论 |
| [C4 agent / baseline / 2](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-2-baseline/result.json) | C0 仅 lanes 32→20：205.45，暂选；C1 provider 无模型路由，无提案 | C0 沿用错误 unit 数；C1 是外部可用性结果 |
| [C4 agent / semantic / 2](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-2-semantic_contract/result.json) | C0 仅 lanes 32→20：207.35，暂选；C1 停止 | 停止时认识到 M256 的容量风险，但不能据此证明所有小 tile 都更差 |
| [C4 manual / baseline / 0](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-0-baseline/result.json) | C0 K512→256：238.85，暂选；C1 再 N64→128：224.45，暂选 | 两个可测方向有效；“已恢复双缓冲”没有候选活动证据 |
| [C4 manual / semantic / 0](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-0-semantic_contract/result.json) | C0 K→256：234.9；C1 再 N→128：225.95；均暂选 | 与 baseline r0 完全相同的两个候选 SHA |
| [C4 manual / baseline / 1](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-1-baseline/result.json) | C0 按位置缩窄扫描至最多 8 行：186.05；C1 再 K→256：178.45；均暂选 | 大收益来自条件性的候选集合缩减；连续 position 前提仍需契约与 held-out |
| [C4 manual / semantic / 1](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-1-semantic_contract/result.json) | C0 K→256：237.45；C1 再 N→128：220.55；均暂选 | 与 r0 相同 SHA；更快计时不证明找到不同变换 |
| [C4 manual / baseline / 2](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-2-baseline/result.json) | C0 K→256：237.05，暂选；C1 停止 | 识别扫描问题但不愿假设位置连续；是合法域证据需求，不是性能计数缺失 |
| [C4 manual / semantic / 2](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-2-semantic_contract/result.json) | C0 K→256：244.35，暂选；C1 再 K→128：263.6，拒绝 | 进一步消除资源提示并未改善总时间 |

前两个已知 SWA 失败的验证比例、生成物、依赖契约与未决归属，详见 [online_failure_review.md](online_failure_review.md)。此处不重复，也不把其未决问题升级为编译器缺陷。

## 1. 资源提示能定位问题，不能替代资源模型和收益验证

SWA 的 PH-MR-001 明确指向 **Right/L0B**，不是 Vec/UB。删除 identity 乘法的提案，以及部分 K 缩小提案，仍使用“释放 UB 就能容纳第二个 Right buffer”的表述。基础输入已经保留提示中的 `Right` 字样；因此这是对已有信息的错误关联，不能简单归因于缺少计数器。不过，将内存空间标识、受影响 operand 和共存分配组织成易查证的事实，可以降低这种关联成本。

不同结果需要分开判断：

- **stage 2→1**：两个 SWA agent 会话分别测得 323.45、323.75，明显慢于各自 278.9 基线。候选的新 `perf_hints.log` 没有 PH-MR-001，因为已不再请求 depth 2；**警告消失不等于恢复重叠或变快**。
- **manual A/B K→128**：新候选没有 PH-MR-001，且 [MemoryReuse 输出](../../../build_output/pfdb_development/optimization-pilot-v1/swa_manual-0-baseline/attempt-0/capture/passes_dump/37_after_MemoryReuse.py) 的 `proj_a_mm` 在第 159–160 行分配两个不同的 32 KiB Right buffer，第 187–202 行的相邻 K 分片分别引用它们。这至少说明“还是旧的同一缓冲、提示仍在”不能从初始 profile 沿用。它尚未证明设备上流水重叠和端到端收益。
- **manual 仅 A 的 N→64**：新输出仍有两个 Right PH-MR-001；减小源码 N 参数未必按想象减小最终 L0 分片和共存关系。应查生成物，而不是只算源码 tile 元素数。
- **删第二次源码 slab load**：199.05 的搜索结果没有胜出；源码上少一次读，不证明 lowering 后少一次物理搬运。没有指令差异或候选 PMU 就无法把原因锁定为 GM 带宽。

这组结果反驳的是“消掉 hint 必然加速”的启发式，不反驳 hint 本身。pfdb 更有价值的接口应把 **请求深度、实现槽位、内存池、源码/生成物锚点、候选版本** 关联起来，并明确尚缺活动重叠证据。按本轮协议，agent **没有收到**事后保留的新候选提示与分配；对“提示仍在”的断言应批评为无刷新证据的沿用，而不是忽略已经收到的新结果。本轮不能评价自适应 profiling 工作流是否会解决它。

## 2. 生命周期差额支持组织假设，不证明差额全是调度开销

SWA agent 的暂选候选将 gather 的 `parallel + at` 改为一个 SPMD grid，同时将 attention 的 token/head fanout 改为 SPMD。提案引用初始 gather wall 20.68 与 `busy` 5.44，以及路径上的 data/core wait；最终搜索中位数为 249.7，比 278.9 低 29.2。

这是有具体依据、通过冻结 fixture 的组织变换；它也与 [历史 SWA 报告](../../../benchmarks/implementation/deepseek-v4.1-flash/performance-gap-study.md) §5 中提交方式实验的方向一致。但是：

- 本轮同时改两个区域，未收集两者的隔离对照；不能给每次提交分配固定 µs 节省。
- `task_busy_us` 是各 row 的起止包络；`row_duration_us` 是生命周期，均不是 pipe 活动时间。wall 与 busy 的差不能全部归入主机开销，core-wait 也不证明所有核在做有用计算。
- 候选未提供新的 level-4/PMU 对照，无法确认任务提交数、row 数、实际占核、等待和活动时间分别如何变化。

semantic 条件明确给出了上述时间定义；仍出现将 busy 当“计算量”或“实际执行”的表述，说明**文字定义可用并不等于提案能稳定使用它**。这一观察只表明解释风险存在，不能证明语义补充无效；还缺独立、足量、受控条件比较。

相反，SWA agent baseline replicate 2 也做了“32-block SPMD 投影 + 分离量化”，验证通过但为 283.75。更多 block 暴露与更少融合的方向本身不保证收益。该候选还改变 scratch 的自动依赖与 mixed 组织；没有候选 profile 时，不能从总时间判定是块粒度、依赖、搬运还是其他代价抵消收益。

## 3. C4 暂选候选正确找到了可测方向，但停止依据有算术错误

初始路径里 C4 projection 的包络为 115.7 µs，20 个 AIC、32 个 projection lanes。把分块工作量与核数一起考虑，比只问“occupancy 是否高”更具体。三个候选的源码推导如下，均以当前首块 512 行为条件：

| 配置 M/N/lanes | 逻辑 unit 总数 `ceil(512/M)×(1024/N)×2` | 每 lane 的最多顺序 unit 数 | 观察 |
| --- | --- | --- | --- |
| 128/256/32，初始 | 32 | 1 | 32 个 lanes 由 20 个核调度；核上分配需 capture 证明 |
| 64/128/20 | 128 | 7 | 搜索 245.9；把工作更均匀分配未抵消其他成本 |
| 256/256/16 | 16 | 1 | 编译阶段失败，无计时 |
| 112/256/20 | 40 | 2 | 搜索 198.85，暂选 |

计算依据是 [候选源码](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-0-semantic_contract/attempt-0/candidate.py) 第 55 行 `OUT_DIM=2*HEAD_DIM=1024`，及第 167–170 行 `lane_blocks/lane_units/lane_steps`。**lane、循环内 unit 和物理执行 row 是不同对象。** M112 的首次提案正确写出 40 units；停止提案却说成 20 units、每核一次，还称 M128 只有 16 units、4 个核空闲。后两个数字均与代码不符。计时通过不应让错误的解释进入经验库。

replicate 1 两个条件的 C0 都只将 lanes 32→20，源码 SHA 前缀同为 `d1bb37f8aa6305f7`，搜索分别为 baseline 206.55、semantic 206.1。baseline 提案称有 128 units，semantic 提案称有 256 units；源码 M128/N256 实际仍只有 32 units，20 lanes 中最多两次。两边出现同类错误，也得到同一个有搜索收益的代码，不能据此宣称 semantic 条件带来更好的变换。**这些算式所需的源码本来就在输入中，不能将算错直接归因为 pfdb 缺少原始指标。** 它们适合作为下一轮检验对象：将同样事实呈现成明确的 work/grid/core 实体关系，是否能减少错误；不能先据此认定必须新建某种自动源码索引或 DAG 功能。

replicate 2 两个条件再次产生这个相同 SHA，搜索分别为 205.45、207.35；公开解释仍使用“每块 8 units、最忙核 16→13 units”的错误计数。baseline 第二次模型调用遇到 provider 的 `Model unavailable`，不是模型超时、编译或数值失败。semantic 停止提案注意到 M256 accumulator 的容量风险，是正确的风险方向；但其“小 tile 增加流量所以严格更差”的结论仍忽略并行和缓存代价，不能由逻辑流量单独证明。

M112 相比 M128 引入首块 48 行重复工作，尾部 clamp 带来重叠投影；现有验证表明这个 fixture 的 `compress_state` 与 `cmp_kv` 两个输出通过现有比较器，不代表所有后续块、动态长度、历史状态和并发写契约已覆盖。不能据此证明“已到最佳点”。

M256 候选的 FP32 accumulator 按源码为 256×256×4 = **256 KiB**，超过既有 A2A3 Acc/L0C 128 KiB 容量；[候选提示](../../../build_output/pfdb_development/optimization-pilot-v1/c4_agent-0-baseline/attempt-1/capture/report/perf_hints.log) 明确出现 PH-AT-006：结果超 L0C，当前结构未被自动切分。最后保留 pass 为 `37_after_MemoryReuse.py`，终端只有不透明异常。可以认定容量/自动切分风险已有直接证据；**不能在缺少实际异常的情况下断言这就是唯一失败原因，更不能称编译器有 bug。**

其“96 MB→64 MB 流量”的提案估算还需要标为逻辑加载量。缓存复用、L1/L0 搬运与 HBM 流量不是同一口径；既有报告也未授权用源码读次数当作实测 HBM 带宽。

replicate 1 semantic 的 C1 将归一化结果从 `ntsp` 的 GM store/load 改为直接 on-chip slice，并提升两个模板 load；baseline 的 C1 同时把 RMS 从 16 行展开改为直接处理 1 行，并删除 GM staging。公开提案的数值恒等性理由不能单独证明 tile 布局、切片和寄存/缓冲生命周期合法。两者保留日志都只有 compile start/done，无 pass dump、capture 或实际异常，进程返回 1；因此只记录“失败且原因未明”，不赋予编译器、布局或精度根因。它们与历史案例的直接 tile 切片风险相呼应，但不是那个失败的重现证明。

## 4. C4 manual 同时暴露了有效变换、未证实机制和合法域需求

五个会话首先提出 K512→256，均通过冻结 fixture，得到 234.9–244.35 µs；源码 SHA 前缀均为 `c7ea6e9240a405e8`。三个会话随后加 N64→128，源码 SHA 均为 `69a7f9d9581a1136`，搜索分别为 224.45、225.95、220.55。两种条件多次提出相同代码，不能把这些不同计时当成信息条件导致了不同优化能力。

K→256 后总时间更低，支持“这个 K 配置值得保留和确认”；**不确认“所有 stage 已双缓冲”**。事后候选日志显示：初始六条 PH-MR-001 分属 Left/L0A 与 Right/L0B 各三条；K256 后 Right 三条消失，Left 三条仍在。N128 后提示增至九条，含 Left 三条和 Right 六条，但总搜索时间继续降低。最后一个 semantic 会话再改 K128，提示全部消失，总时间却从 244.35 增至 263.6。这是更直接的“资源提示不能代替目标函数”反例，也说明源码 K 变小不代表所有 pool 的最终分片同时按同一比例缩小。相关候选日志见 [K256](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-0-baseline/attempt-0/capture/report/perf_hints.log)、[N128](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-0-baseline/attempt-1/capture/report/perf_hints.log)、[K128](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-2-semantic_contract/attempt-1/capture/report/perf_hints.log)。这些新日志没有反馈给模型，公开提案的机制确定性超出了当时可见证据。

baseline replicate 1 选择另一条路径：源码内每个压缩输出扫描 512 行、最多只有 8 行贡献，提案将其改为位置窗口推导的 `pool_lo/pool_hi`。搜索从 249.35→186.05，再叠加 K256 到 178.45，是本轮值得跟进的较大改善。它同时保留 mapping 和位置判断，并显式写出“contiguous prefill position_ids”前提，比只根据高 scalar 指标盲目加核更接近可执行的算法变换；与 [历史 compressor 报告](../../../benchmarks/implementation/deepseek-v4.1-flash/compressor-ratio4-performance-gap-study.md) §5.1 的条件性扫描缩减相符。

该候选的 [diff](../../../build_output/pfdb_development/optimization-pilot-v1/c4_manual-1-baseline/attempt-0/candidate.diff) 使用 `position_ids[tile_base]` 推导物理 token 下标，只有相应连续性关系成立时才能保证候选集合不漏项。冻结搜索 fixture 的 initializer 为 `arange`，两个输出都通过当前比较器；**尚不能把它写成任意合法 position/mapping 上的等价变换**。另一 baseline 会话识别同一 scalar 扫描问题，却因连续性未被证明而停止。这一对照指出明确的信息需求：输入域与可依赖前提需要权威契约及验证锚点；单纯补更多性能统计无法裁定“利用结构”还是“错误收窄语义”。也不能反过来把谨慎停止当作优化能力缺陷。

扫描缩减后的公开提案称 projection 已变为最大瓶颈；在没有新 profile 的本轮只能视为合理待验证假设。端到端时间减少不能确定新阶段排序，更不能由旧 profile 的阶段时长扣减得到精确剩余成本。

## 5. 小收益和正确性必须与搜索选择分开

manual A/B K→128 的相同源码 SHA-256 前缀 **`c6e217bec89e36f7`** 在四个会话中分别得到 199.7、200.55、200.2、194.7。最后一个被暂选，另外三个被拒。这里不能断言波动的具体来源；但同一代码出现这组结果，已足以阻止把 195.7→194.7 的单次搜索优势归因于 semantic 信息或“真正实现双缓冲”。独立交错确认、效应阈值与等价口径必须先完成。

两个 M16→M8 SWA 候选均未获得验证或计时，pass dump 都停在 `11_after_OutlineClusterScopes.py`，终端保留 `ERR99999`、stderr 无具体诊断。提案将 8 个有效 token 对应的 padding 理解成“减半实际 MAC”；但物理 fractal、合法 tile 形状、切分和 padding lowering 仍待查证。此处应保留“编译阶段原因未明”，不能仅由阶段位置编造 alignment 报错。

数值已失败的 identity 删除候选、只有不透明编译异常的候选、模型超时及 provider 无路由，是不同失败类别。它们应分别进入合法性覆盖、诊断完整性、模型预算和外部可用性指标，不应合并成一个“未加速率”来解释 pfdb 质量。

## 6. 能归给 pfdb 的问题，与仍属未知的问题

| 现象 | 当前可归因到哪里 | 对 pfdb 的具体启示 |
| --- | --- | --- |
| 输入已写 Right，提案改 Vec 临时量来“腾容量” | 已有事实被错误连接；不证明缺少 profiler 数据 | 内存池与分配实体绑定；让错误跨池推论易于发现 |
| semantic 已定义生命周期，提案仍写计算活动 | 公开解释未完全遵守口径；不能据此量化条件效果 | 输出字段附短定义、不可相减/不可换算边界；同时检验呈现和信息本身 |
| 对旧 profile 的 PH-MR 继续宣称“仍在”或“已恢复” | harness 明确没有回传新提示；提案无新证据却沿用/确认机制 | 每条事实绑定 source/build，跨版本引用显式过期；本轮不评价完整自适应查询流程 |
| unit、grid、提交、row、核数混用 | 可核验的程序组织推导错误 | 提供实体关系与可追溯计数，避免只有一个含糊 task 数 |
| 数学恒等改动改变跨核交接 | 变换合法性未证明；归属尚未定 | 关联放置、数据交接、同步与验证证据；数据库不签发代码合法性证明 |
| 同一扫描问题，一个会话利用位置连续性，一个会话因未证明而停止 | 合法输入域和特化前提没有得到共同、权威的裁定 | 关联输入契约、条件与测试覆盖；避免把 fixture 值模式提升成普遍规则 |
| M8、projection split 不透明异常 | 外部执行/编译诊断不完整 | 支持导入原始异常和最后成功 pass；更多利用率统计解决不了该缺口 |
| 相同 SHA 搜索中位数跨过接受线 | 确认不足，具体波动来源未知 | 将搜索暂选和独立确认分字段保存，保留所有样本/配置/失败 |
| 两个模型超时 | 执行预算结果；信息量/难度是否导致超时未知 | 保留超时和资源成本；需要独立呈现消融才能归因冗余 |
| 一次 provider 无模型路由 | 外部服务可用性，未生成候选 | 与推理预算、内核合法性和 pfdb 质量分别计数 |

后续最有辨识力的工作是：为暂选候选完成独立确认及 held-out 验证；由外部 harness 采集新候选 level-4、生成物与必要的 PMU；将新旧事实按 source/build 对齐后检查机制。对于不透明失败，先补异常和最小合法性诊断。**本次复核没有证实语义补充改善搜索能力，也没有证实某类信息冗余；固定证据包尤其不能评价 PMU 或自适应查询是否有用。它提供的是下一轮信息/呈现消融的具体错误类型与可审计案例。** pfdb 仍只分析和组织已有工件，不承担采集或修改 runtime 的职责。
