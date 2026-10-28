# pfdb 接口取舍的独立审查

日期：2026-10-21。用途：为[开发计划](../../../docs/debug-and-tune/pfdb-development-plan.md)提供审查输入；**不是整个计划的完成报告，也不是删除授权**。本审查只读取源码、已有材料和已完成实验结果，没有修改实现、数据库、capture 或上游源码，没有启动模型调用或设备任务。

## 1. 当前证据允许什么决定

目前足以确认的是访问修复和少数描述/口径问题；不足以确认的是哪些底层分析应删除，或哪种全新默认目录能提高真实优化收益。应先修正“工具承诺回答的内容”与实际事实的差异，保留已有能力，再分别实验发现方式、表达方式和新增信息。

- 任务 name 查询与 query 选择器碰撞、MCP 截断无法提高预算，已有确定性修复及公开入口测试，属于可达性缺陷，不能计作新增 profiling 信息的收益。[API](../../src/profile_db/api.py)、[MCP](../../src/profile_db/mcp_server.py)、[传输测试](../../tests/test_query_transport.py)
- 修订后的两题表示校准 v5 有 12 次尝试、11 个正常响应全部通过事实/引用检查，另一次是 `provider.no-route`。排序题 facts/columns 提示为 3,712/3,804 B，列式更大；多 row 题为 8,065/4,238 B，列式更小。这说明压缩效果依赖数据形状；没有观察到准确率差异不等于已证明非劣，更不能证明优化收益。数字来自忽略目录中的 v5 protocol/results；版本化材料见[题目](corpus/cases.json)和[执行器](run_offline.py)。v4 的题意歧义不应与 v5 合并评分。
- 交互式单题说明对照 `discovery-code-v2` 中，current/explicit 各三次均答对数值和布尔值；两组客户端 wrapper 调用总数同为 15，响应文本总量分别为 32,994/62,565 B。仅修改 `why_long` 描述没有显示稳定的成本优势。该评分不覆盖所有自由文本推断或引用语义，样本也只是一题。原型见[描述干预](mcp_experiment.py)与[交互执行器](run_interactive.py)。
- 本次成功交互走 Code Mode；一个客户端 `execute` 可以执行多个 pfdb 调用。41 个 MCP schema 不能直接解释为模型每次加载 41 个原生工具，也不能把 wrapper 调用数等同数据库查询数。后端追踪已记录独立的查询/响应摘要。[追踪实现](mcp_experiment.py)

这些结果支持“保留基线，继续做局部信息关联对照”，不支持“统一改成列式”“某个新描述已经获胜”“41 个接口都应压成三个”。

## 2. 逐接口建议

表中“保留”表示当前路径继续可用，不代表其默认曝光已被实验证明最优；“按需候选”表示下一轮实验因素，尚不改变生产默认目录。所有项目的复审触发器都是 runtime/model/channel/workload 改变、关键证据丢失或新的正向/反例实验。没有任何项目满足本轮直接删除底层实现的证据门槛。[接口清单](interface_inventory.csv)

| 接口 | 建议状态 | 已有独立价值或问题 | 仍需评估的替代/反例 |
| --- | --- | --- | --- |
| `list_runs` | 保留 | 选择 run/rank；不是单 run 摘要的重复 | 大工作集首层字段是否过多；不丢 trial/rank |
| `overview` | 保留；合并呈现候选 | 同时给 bench/topology/graph size | 与 inventory/bench 摘要组合是否减少导航，不隐去缺失 |
| `inventory` | 保留；详情按需候选 | artifact 状态、来源与 reanalysis 是独立问题 | 正常导航只给状态，完整 hash/function provenance 按需 |
| `density` | 保留 | 发现未知时间窗口 | 不能因已知 task 单题不调用就删除；有多个热点的反例 |
| `sparse_regions` | 修复描述；保留 | 当前只排序稀疏 storage bands，不给 blocker | 与 density 分组发现是否更易用；两种 band 坐标仍须区分 |
| `why_sparse` | 修复描述；保留 | 官方 ready/idle/overhead 积分 | 这些计数不足以单独证明派发瓶颈；保留单位和计数域 |
| `region` | 修复描述；保留 | 指定窗口中的 task/gap 列表 | 不能把窗口相关性包装成“为何空闲”；尾部/交叠场景 |
| `core` | 修复描述；保留按需候选 | 单核 rows/gaps | 当前不回答 other cores；跨核问题应转 region/idle_window |
| `idle_window` | 保留 | producer-end→consumer-start 的另一引擎占用 | 与任意 region 不是完全重复；默认下一 CP task 的边界须可查 |
| `task` | 保留；局部关联呈现候选 | task 身份、时间和 CP membership | 与 why_long 组合不应丢 path/identity，也不自动加载全部 rows |
| `tasks` | 保留；入口已修复 | family/name→task ID 的可靠定位 | 多实例排序/摘要能否少查几次；保持完整实例可达 |
| `deps` | 保留 | direct/reduced/dataflow/wait 回答不同图问题 | 不能把 transitive reduction 当作任意删边证明；WAIT 正向场景 |
| `subgraph` | 保留按需候选 | 有界邻域、external nodes 和 capped 状态 | 不是整个 deps 或完整 CP 替代；分叉图正向例 |
| `why_late` | 保留 | data-ready/FIN/dispatch/receive/start 的分段 | 与 CP gap/stall 定义独立；保留 early-dispatch 有符号值 |
| `why_long` | 保留；语义澄清 | 已有 min/max row 与 family rank | even-n median 定义需明示或修复；不能当 per-core 算术耗时 |
| `rows` | 保留；明细按需候选 | 完整物理行时序和极值/尾部 | min/max 不足以替代启动分散和尾部；预算恢复必须继续有效 |
| `scheduler` | 保留；全量 phase/row 按需候选 | 官方 scheduler 摘要及局部 phases | task-local 与 run-wide views 的字段/范围须一致说明 |
| `early_dispatch` | 保留 | eligibility 与 timestamp proof 的区别 | 不被负 gap 或调度参数替代；partial/unavailable 反例 |
| `pmu` | 修复描述；保留 | 样本、counter 和分母比例 | busy ratio 不能证明峰值吞吐；独立 capture 绑定是另一个缺口 |
| `critical_path` | 保留 | 官方 Observed/Static path/contribution | 不应被局部 why_late 或高 occupancy 取代；新路径结构正向例 |
| `perf_hints` | 保留；局部索引候选 | requested 与 realized buffer 的独立证据 | 只过滤“常见 hint”会漏稀有 alias 反例；必须能取全 |
| `memory` | 保留 | buffer 空间、限额与 placement | PMU 不能替代编译期容量；无该文件不等于零占用 |
| `incore` | 保留按需候选 | 重叠 pipe events/metrics/export 状态 | 真实两例未充分评估；已有 synthetic 正向测试不证明加速 |
| `args_dump` | 保留按需候选 | 物理 shape/stride/offset/values | 不可用逻辑 TensorSpec 替代物理切片；合法页边界正向例 |
| `scope_stats` | 保留按需候选 | ring/site 峰值与容量生命周期 | 不能用 kernel memory 代替 scope/ring 资源；配对/截断反例 |
| `bench` | 保留；raw rounds 后置候选 | 独立无插桩目标，含 median | overview 仅摘要不能替代独立 strata；保留 raw 可达 |
| `host` | 保留按需候选 | PID/invocation/CPU/off-CPU/cold-warm | 两个 device 算子不足以判冗余；host delay/loss 正向场景 |
| `attach` | 保留；维护目录候选 | 注册已有辅助工件；不采集 | PMU-only 目前不被 discover_optional 发现；绑定身份先于自动 join |
| `refresh` | 保留；维护目录候选 | 显式重算，保留旧事实的语义锚点 | 普通 query 不可偷偷替代；失败 rollback/changed hashes |
| `compare` | 修复/增补契约；保留 | 当前 capture gate 和数值 delta | `compatible` 不能被解释为同 workload 合法优化；见第 3 节 |
| `baseline_diff` | 保留 | named baseline 导航复用 compare | 不是第二套分析；需沿用相同 contract/objective gate |
| `baseline_list` | 保留按需候选 | 稳定 baseline 定位 | 与 list_runs 组合发现可测；不能漏 prune-protected baseline |
| `baseline_add` | 保留；记忆/维护按需候选 | 命名与 prune 保护 | 仅记录名称不证明验证通过；默认 mean 与本例 median 区别 |
| `register_trial` | 保留；增补记忆原型 | parent/hypothesis/changed_files | typed outcome、适用域与 diff 原型先在外部包测试 |
| `bind_trial` | 保留；记忆/维护按需候选 | 显式 run 绑定 | 不能按相同 task ID 猜实现；验证身份与 capture 身份区分 |
| `attach_bench` | 保留；记忆按需候选 | 无 profile run 时仍能登记 bench | 摘要不等于独立 acceptance；必须知道 estimator/validation |
| `set_verdict` | 保留；修正文档口径 | 保存 caller 的决定 | 不是 pfdb 数值验证；当前无 precision/runtime-error 类型 |
| `list_trials` | 保留；紧凑历史候选 | 结果与 lineage 记忆 | completed trial 的 run 可被 prune，不能依赖失效 locator |
| `note` | 保留按需候选 | 自由文本备注仍有低成本用途 | structured outcome 不必消灭备注，但不能只靠备注做安全比较 |
| `render` | 保留按需候选 | 图像/HTML/Perfetto 关系与跨模态证据 | 导出有文件副作用；多模态用途和占用查看需独立检验 |
| `version` | 保留；客户端自动探测候选 | contract 协商/升级锚点 | 可由初始化信息承接默认展示，旧调用仍可保留 |

来源分组：[Z0](../../src/profile_db/query/handlers_z0.py)、[Z1](../../src/profile_db/query/handlers_z1.py)、[Z2](../../src/profile_db/query/handlers_z2.py)、[Z3](../../src/profile_db/query/handlers_z3.py)、[Z4](../../src/profile_db/query/handlers_z4.py)、[modalities](../../src/profile_db/query/handlers_modalities.py)、[runtime views](../../src/profile_db/query/runtime.py)、[MCP lifecycle/render](../../src/profile_db/mcp_server.py)。

## 3. 源码可确认的语义与接口风险

### 3.1 工具说明超出 handler 能回答的范围

四个可直接修正的例子：

1. `core` 的 owner question 问“其他核在做什么”，但两段 SQL 都带 `core_index = params.core`，仅返回目标核 rows/gaps。说明应改成单核时间线，跨核查询另给明确路径。[Z2 core](../../src/profile_db/query/handlers_z2.py)
2. `sparse_regions` 问“每段 blocked on 什么”，但返回字段只有 band/window/engine/core occupancy，未查询依赖或 blocker。应声明其定位作用。[Z1 sparse_regions](../../src/profile_db/query/handlers_z1.py)
3. `why_sparse`/`region` 说明问“为什么”或二选一瓶颈，而当前事实分别为官方计数积分、窗口活动/gap。积分单位包含 task-us/core-us/elapsed-us，不能混加，也不提供因果标签。[Z2](../../src/profile_db/query/handlers_z2.py)、[counter_window](../../src/profile_db/query/runtime.py)、[runtime 指标台账](../../../docs/debug-and-tune/pfdb-runtime-integration.md)
4. `pmu` 问“哪个 pipe 接近 saturation”，实际输出是每个 counter 的 `SUM(value)/SUM(total_cycles)`，并保留样本。比例可以帮助提出机制假设，但没有 bytes/FLOPs/实际峰值，不足以证明达到了 HBM 或算术吞吐峰值。[Z4 pmu](../../src/profile_db/query/handlers_z4.py)

这类修复不需先证明带来设备加速；需要的是说明与现有事实一致。变更应在新的实验基线冻结，不能中途修改对照条件。

### 3.2 `family_median_us` 是 lower median

`why_long` 排序后取 `vals[(len(vals)-1)//2]`；偶数样本例如 `[6,10]` 返回 6，而不是中间两数平均值 8。现有 golden QA 明确固定了该输出，说明这是已传播的行为，不应悄悄改值然后沿用旧解释。[实现](../../src/profile_db/query/handlers_z4.py)、[快照](../../tests/golden_qa/questions.py)

建议先明确这是 lower order statistic，或单独修成常规 median 并更新版本/快照/手册；同时测试 odd/even/ties。该修复是 pfdb 的局部统计，不应归因于 runtime。此审查没有修改值。

### 3.3 capture compatibility 没有覆盖 workload 与 validation

`compat_reasons` 检查 program、level、clock、core count/topology；另外比较 critical-path/scheduler 的分析身份。它不核对 workload shape、position/mapping、reference/comparator/tolerance、输入快照和完整目标 estimator。family 聚合还是字符串对齐与 task span 求和。[compare](../../src/profile_db/lifecycle/compare.py)

这不意味着当前“描述性 delta”算错；问题在于无范围限定的 `compatible=true` 容易被当成同 workload 的优化许可。应区分描述性、优化和机制比较，并对未知契约返回 unknown；不能把旧库缺失信息迁移成 PASS。当前 bootstrap 指标明确是 `bench_mean_speedup`，不能确认本例 median 目标。[bootstrap](../../src/profile_db/lifecycle/bootstrap.py)

### 3.4 部分缺口位于绑定与记忆，不是计数器缺失

- initial ingest 可以读 capture 自带 PMU，而 `discover_optional` 的附件清单没有 PMU。单独 PMU-only capture 的安全关联应先验证 build/workload/pin/采样域；添加文件名识别后直接按 task ID 合并是不安全的。[attach discovery](../../src/profile_db/ingest/analysis.py)
- `set_verdict` 仅有 win/neutral/regression/compile_error，没有数值验证记录；verdict 是 caller 提供的字符串。它也不检查某个 win 是否有 PASS 的 implementation/validation 绑定。[trial](../../src/profile_db/lifecycle/trial.py)
- trial 行会保留，但其已完成 run 不在 active-trial 保护集合中；`list_trials` 通过 left join 回退读取 run 的 bench，run 被清掉后该回退值会消失。应保留足够小的条件化实验结果，而不是无条件保留所有 traces。[working set](../../src/profile_db/lifecycle/working_set.py)、[prune](../../src/profile_db/lifecycle/prune.py)、[trial](../../src/profile_db/lifecycle/trial.py)

### 3.5 旧说明与当前 runtime 权威口径分离

`profile_db/DESIGN.md` 的 6.3、idle-gap 分类和若干 handler 模块注释仍描述历史归因规则，当前公开 runtime 台账已说明 gaps 主要是 occupancy 事实、why_sparse 使用官方计数。清理应更新或明确归档旧设计的适用版本，避免 agent 读到两套冲突口径；不应重新实现旧规则。[旧设计](../../DESIGN.md)、[当前台账](../../../docs/debug-and-tune/pfdb-runtime-integration.md)

## 4. 最小后续实验：优先检验两类缺口

### 4.1 比较/验证契约

用外部、版本化证据包加信息，不先迁移 schema。基线给现有 run/bench/profile facts；处理组仅增加有来源的 workload、reference/comparator、validation 覆盖和 objective。任务含同 workload、T=512 对 T=8192、不同 position/mapping、validation 未知、mean/median 混用、独立 PMU 身份未知。

评审检查能否区分“描述性数字”“可接受优化”“机制线索”，不要求复述某个修复。正向对照保留一个完全可比的有效候选，防止 agent 通过一律拒绝获得高分；移除对照分别去掉 workload、validation、objective，不能一次全删后把贡献归给每项。

### 4.2 指标语义和 task/row 关联

先分开做两种干预：

- 纯表达：同一 facts 集，只将 task、LONG、rows 组织在同一 bounded 对象下并保留稳定 evidence ID，确定性验证逐字段可逆。不得偷偷加入新的 launch-spread 或结论。
- 新派生/定义：明确 busy envelope、row lifecycle、dispatch→FIN wall、pipe-active 不可由时戳证明；另加 launch spread/row 分布时单列派生条件，不称为纯压缩。

正向场景包括长 envelope/短 rows、单长尾 row、所有 row 同时开始但等待很长、MIX 两引擎、早派发负分段。rows 与 CP 联合或可替代程度，应分别移除一种并提供需要它的反例，避免“相关信息互相替代”被判成全都无用。

这两类优先于通用 IR 解释器、更多 counters 或新增万能入口：前者分别对应已观察的比较误读和生命周期/算术混淆，后一批需求尚无独立收益证据。

## 5. 未完成实验与放行条件

| 未完成项 | 所需最小证据 | 当前禁止据此做的决定 |
| --- | --- | --- |
| 10 题完整预试 | 每题 3 次匹配重复、相同模型/预算/证据访问；逐题盲评 | 两题读数成功不等于机制/合法变换成功 |
| 平铺/按需发现 | 固定真实客户端模式，正向覆盖维护与低频入口 | 不能按 schema 数就改目录，也不能用无工具会话判接口失败 |
| 有界关联摘要 | 可逆信息审计、极值/稀有证据可达、恢复成本 | 不直接给生产新增 brief/inspect/万能 tool |
| 两类信息增补与移除 | 单因素正向和消融，明示原始/派生/索引 | 不把额外定义或映射收益归给格式 |
| 低频能力 | host 延迟、scope 峰值、物理 args、incore 重叠、rank/clock/epoch 正向用例 | 没有真实数据时保留，synthetic 仅证明语义和可达性 |
| 新合法输入与 holdout | 原 spec 内反例、至少两类工作结构的外部算子 | 不把固定连续 fixture 的 PASS 推成通用合法性 |
| 在线优化确认 | 相同外部 harness/黄金/预算，全部候选含失败，独立确认 incumbent | 不用离线答题/工具调用下降替代最终验证和完整 latency |
| 集中清理/迁移 | 逐项替代等价、旧调用与旧库、副本升级回退、runtime parity | 不因用例少删除底层，也不改上游源码或重解释旧事实 |

已有[运行时集成测试](../../tests/test_runtime_integration.py)为 scope/args/host/incore/rank/epoch 等提供了现实格式的 synthetic 正向场景；它们可复用于发现实验，但不能被写成新增的真实设备收益。继续复用[唯一 runtime 台账](../../../docs/debug-and-tune/pfdb-runtime-integration.md)，不要再建一份上游函数清单。

本审查未运行测试、未逐条审查所有模型自由文本、未执行新优化或确认统计。风险是当前小样本只能排除少数设计误判，不能据此宣称接口等价、冗余或 pfdb 已提高优化成功率。
