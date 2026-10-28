# pfdb 开发与实验记录

本记录跟踪[后续开发计划](pfdb-development-plan.md)的实际执行。状态：**本轮实施已闭环**。完成主静态研究 150 次尝试、24 会话在线预试及独立确认，发布工作树为 pfdb 0.5.0；未发布软件包或提交 Git。访问修复、表示预试、真实优化和发布检查分别记账。

**结果支持修复可靠访问、缺失值/指标口径和实验记忆的确定性缺陷；不支持删除底层能力或认定新的统一目录更优。** 在线处理组对基线的四起点等权配对比为 0.994×，不构成总体优化能力提升，也不足以断言处理组更差。最大的 fixture 收益仍依赖未证明的 position 连续性前提；因此时间变快、解释正确和通用合法性必须分别记录。[逐会话公开摘要](../../profile_db/research/pfdb_development/optimization_summary.json)

## 基线与工件

- A0：pypto-lib `d6cbf967bcf0d4b4bc412467cf28c944e26b1362`；pfdb 0.4.0，DB schema 8，MCP schema 4；27 个 query、41 个 MCP tools、8 个写操作。
- A0 CPU 测试：288 passed、11 skipped。跳过项不当作已验证能力。
- 保存了 A0 代码归档、工具/schema 清单、公开入口失败、响应规模与原始数据库 hash，位于忽略的开发工件目录。
- compressor 数据库有 7 个 run，其 35 个注册工件均可解析到当前文件。旧 SWA 数据库有 2 个 run，但 10 个注册工件路径全部缺失；另一个 SWA 数据库为空。不能把旧库中的统计当作可重新分析的完整 capture。
- 既有 compressor 报告与资产保持原样；不改原数据库、原 capture 或 PyPTO/runtime。

## E0：已完成的公开入口修复

1. Python API 的 query 选择器与 task `name` 筛选分开，保留原 `query(name="overview", ...)` 写法；CLI/MCP 不再因同名关键字碰撞失败。
2. 所有注册 query 的 MCP schema 增加 `budget_bytes`，沿用已有 query budget 与同一 handler。截断后可按提示提高预算，重新取完整前缀，替换旧响应。
3. MCP 的预算/参数验证错误转换为结构化 `QueryError`。没有引入分页游标、修改 runtime 分析或新增 profiling 事实。

新增公开路径测试验证：64 行响应从默认截断恢复到完整，API/CLI/MCP 字节一致；原数据库 hash 不变；非法预算被拒绝；task name 筛选和旧 API 写法兼容。首次定向验证为 11 passed；完整 A1 回归为 301 passed、11 skipped。

随后将同一预算恢复能力补到只读 lifecycle 工具（compare、baseline diff/list、trial list），并验证长 trial 历史可以完整取回；相关 API/MCP/传输测试 17 passed。写操作没有增加重试接口，避免重复执行写入。

本轮采用预算重试而非游标续取：它是一次新读取，不承诺跨 refresh 拼接旧片段。此选择仅修复可达性，不预先决定最终默认响应或工具分组。

## 表示校准与 OpenCode

OpenCode v2.1.2。`opencode/deepseek-v4.1-flash` 返回 402 额度错误，免费提供方返回 429；`opencode-go/deepseek-v4.1-flash` 已成功返回 smoke answer。实验固定使用成功的 Go 提供方，不将不同提供方结果混报。

先从完整 compressor 捕获导出两个校准题：完整 median 与局部 projection 排序、长 task 跨度与 physical rows。每题包含确切的 query/selector、稳定 evidence ID、原始事实与来源 fingerprint；未读取研究报告答案，也未用报告重建原始 capture。

校准比较原始逐项 JSON 与可逆的列式 JSON；自动测试确认两者字段、值、顺序、单位与 evidence 完全相同。它是**非交互的表示预试**，不能证明默认工具目录更好，也不是完整 10 题或 24 会话优化评测。

启动预试暴露了 OpenCode 的项目配置解析差异：仅设置子进程 `cwd` 时，`run --standalone` 未找到自定义 agent；同一预验证工作区中同时设置匹配的 `PWD` 后，实际模型调用成功。只初始化 Git 不能充分解决此问题。启动器改为在一个预验证、禁止所有工具的隔离工作区中开启独立新会话，并同步 `cwd/PWD`、校验权限；原失败完整保留，不算作模型解题失败或表示效果。

OpenCode V2 的 agent/system/permissions 配置以[官方配置说明](https://opencode.ai/v2/docs/config)与[agent 说明](https://opencode.ai/v2/docs/agents)为准；不依赖旧字段的隐式转换来推定隔离已经生效。

首个实际完成的表示校准（v4）有 12 个独立会话、无 provider 错误。排序题两种表示各 3/3 通过；该题列式表示反而从 3,622 B 增至 3,714 B。多 row 题从 7,879 B 减至 4,052 B，说明压缩收益取决于记录结构，不能统一切换所有响应。

该预试同时发现题目缺陷：`task_span_us` 未说明是 AICore envelope 还是 dispatch→FIN，多个回答合理地使用 wall_us，不能据旧评分宣布 agent 不会读数；“evidence ID”也未明确区分 query 与 fact。另有回答将 receive→finish 或 busy_us 称作 pipe-active/active union，提示需检验指标语义的表达。已保留原结果，修订题目明确询问 busy_us、wall_us、其语义与事实级引用；新旧协议分开，不事后改分凑出赢家。

修订后的 v5 校准共 12 次尝试，11 个有效回答均通过数值、类型和事实级引用检查；另一次为 `provider.no-route`，作为基础设施缺失保留。不能将 11 个有效回答的通过率当作优化收益，也没有发现列式表示统一优于原表示的证据。

外部执行器已重新采集两版 SWA 的 level-4 工件，均在同一 frozen golden 下通过原 comparator；导入独立工作库后，agent 版有 51 个逻辑 task / 178 个 physical rows，手写版为 30 / 234。真实 capture 与官方读取器的定向回归为 3 passed、4 skipped；跳过项依赖另一模型的固定工件。没有由这些 profiling 数字推导新的无插桩加速比。

## E1：交互导航预试

真实 MCP smoke 中，模型在默认 rows 响应截断后，使用新增 `budget_bytes=32768` 成功取全，得到正确的 busy_us、wall_us 和 max_row_us。这证明恢复路径可被实际客户端使用；单个 smoke 不证明总体优化能力提升。

OpenCode 默认 Code Mode 将 MCP 工具包装在 `execute` 中，一次调用可以批量查询。因此区分客户端调用、实际 pfdb 查询、工具目录获取和响应字节；不能将 41 个后端 schema 直接称作模型每次看到的 41 个原生工具。直接暴露模式的原型没有成功发出后端查询，原因未定，这些会话不用于比较工具说明优劣。

固定 Code Mode、权限、数据、schema、handler 和预算，仅改 `why_long` 的问题描述，完成 3 个重复编号 × 2 条件的交互预试。两组数值/布尔答案均为 3/3 正确；客户端调用各 15 次，后端查询为 17 对 15。改描述后的累计客户端响应文本反而更多（约 62.6 kB 对 33.0 kB）。此处只评分请求的字段，尚不代表自由解释和引用完全合格。

结论是**没有足够证据将该说明原型设为默认**。保留当前实现，继续检验关联、语义和契约信息，避免把减少一两个查询当成系统收益。原协议、全部会话和后端 trace 均保存在忽略的实验目录；重复编号不是服务端可复现随机种子。

## 如何读后续结果

以下分别记录十题筛选、能力正向场景、指标定义、实验记忆与完整优化闭环。未达到门槛的方案保留为未决项，不以“未调用”推断冗余；未实施的候选及下一轮触发条件进入决策台账。

目前没有删除底层能力，也没有宣称 pfdb 已提高算子优化成功率。静态答题、确定性缺陷修复、搜索中收益与独立确认分别报告。

## 十题筛选：表示与信息分开

从已有数据库、源码和生成产物导出 10 道开发题，答案索引只供评审，模型仅见问题和选定证据包；路径作等量脱敏，原文件不变。冻结的问题覆盖提交/grid、lowering form、完成边界、实际 buffers、局部/整体排序、扫描合法性、task/row、occupancy、capture 身份和实验记忆。

| 条件 | 尝试 / 正常回答 | 字段与既定引用检查通过 | 累计提示字节 |
| --- | --- | --- | --- |
| D0 原 facts | 30 / 30 | 30 | 123,351 |
| D0 可逆列式 | 30 / 30 | 30 | 91,518 |
| D1 原 facts＋逐题声明的信息 | 30 / 30 | 30 | 152,478 |

列式在这套材料中减少约 25.8% 提示字节，但两种表示都达到该题集的准确率上限。不能据此证明统计非劣、减少真实客户端 token，或提高优化收益。D1 额外提供了特定生成 form、reference 选择依据和 workload 身份；D0 正确回答 unavailable 不算错，因此“新信息恢复”与准确率分别记账。

这不是统一信息组合的因果估计：每题补充的信息类别明确不同，且 pipeline/ranking 为已有证据足够的负对照。实际研究数据库没有 trial 历史，不能用第十题推断实验记忆是否有用。

逐题的信息恢复如下。每题 D0 两表示共 6/6、D1 共 3/3 通过既定字段/引用检查；这里检查的是表中明确问题，不覆盖全部自由解释或下一次变换质量。

| 题目 | D0 已能正确回答的边界 | D1 实际恢复的信息 |
| --- | --- | --- |
| submit/grid | 8 个逻辑 task、每 task 1 block / 3 physical rows；不证明合并提交 | 核心答案未变；额外生成编排用于查证 |
| lowering identity | 精确 axis unavailable；不能泛化所有 scatter | 恢复具体 `SCATTER_ROW` form |
| completion boundary | 8 个直接投影生产者；无证据删活边 | 核心答案未变；不自动签发依赖修改许可 |
| realized pipeline | 请求 depth=2、实际 1 buffer；独立收益 unavailable | 负对照，无需增加信息才能答本题 |
| local/global ranking | 整体 run1 更快、投影 run2 更短 | 负对照，已有证据足够 |
| bounded scan | 通用正确性未证明；reference 选择依据 unavailable | 恢复 position + mapping 依据，仍不证明 scan-8 通用合法 |
| row/task span | 1086.8 / 1143.6 / 109.08 µs 分属不同范围；非 pipe-active | 核心答案未变；本题到顶未暴露后续 holdout 的误解释 |
| occupancy objective | 占用不证明收益；无无插桩 median 可接受 | 仍不制造未采集的目标值 |
| capture identity | 未证明同 workload；task ID 单独 join 不安全 | 恢复 token 数不同这一身份事实，安全结论不变 |
| experiment memory | 实际历史为空；caller win 不证明 validation | 语义澄清，仍没有历史数据；后用独立 synthetic 历史检验 |

D0 的正确拒答不记为错误。D1 恢复三类具体信息不等于提升答题准确率，更不等于已证明每类信息应成为新的默认字段；其额外上下文、采集/维护与关联成本仍需独立比较。

首次启动在调用模型前被预检拒绝：OpenCode 曾返回不含目标 agent 的目录。保留失败日志；随后对只读目录预检限定最多三次，记录每次结果并继续核验 deny-all 权限。该处理不增加模型重试，也不将启动器失败计成答题失败。

## E3：能力正向场景与两个真实 holdout

预先选定 RMSNorm（512×512 FP32）和 matmul（256³ FP32），分别代表归约和 GEMM 结构。外部执行器各运行一次 level-4，原 specs/golden/tolerance 均通过。两者包含 8 / 16 个独立 task、同样数量的 physical rows，依赖边数均为 0。

这暴露了 pfdb 将合法 `edges=[]` 拒绝为“不是非空列表”的错误。修复仅允许空列表，缺失/非列表仍报结构化错误；未制造依赖边。两个真实 capture 的保存 CP 与 runtime 官方结果完全一致；另有 level-1/4 和非法输入的 6 个确定性回归。

另复用既有 fixture，构造明确标注 synthetic 的 host PID/cold-warm、scope 容量、args 物理 stride、incore 交叠、rank/clock 五个正向问题。两种表示各三次的 30 个回答全部通过既定检查；这些结果只支持能力可解释性，不能证明它们应默认展开或能带来设备加速。

两个真实 holdout 产生 12 个回答，数值字段均正确，但 **5 次把生命周期时间戳误当成 pipe-active 的证明**。原 rubric 还有过严的 overview 引用要求，因此分开报告字段错误与引用覆盖，不修改既有结果以制造赢家。

针对该错误，另注册一个探索性补充条件：相同两个 holdout，仅增加明确的指标定义，共 6 次新会话，全部答对并满足引用要求。该补充是在看到错误后设计，不能当作独立确认或总体错误率估计。它与交互预试中的自由解释错误共同支持一个小范围开发：在现有 `why_long` 中补充 `METRIC_DEFINITION`，无需新建顶层工具或更改 runtime 算法。

新增定义保留历史 lower-middle family median；level-1 不再把缺失 dispatch/FIN 的 wall 显示成有效 0。该限制在共同 query 层覆盖 task/row/path 视图，在 family compare 中明确缺失；原数据库占位值不改。未计时 task 不再计算虚构的 family rank。缺失与定义的确定性测试、既有 golden QA 一起验证。

复审还发现：存在未计时任务的 family，原聚合会把未知值变成零或不完整总和。现按 metric 检查每个成员是否有计时；不完整的 family timing delta 省略并附不可用原因，任务数与其他完整指标保留。真实不存在的 family 仍表示零工作量。该修复属于 pfdb 聚合，不改变 runtime 或原始任务数据。

## 实验记忆与已证实的保留缺陷

使用独立 synthetic 数据库，创建 6 个合法 capture 和 6 个 completed trials，其中一个 trial 显式 attach bench，其余仅 bind run。`keep=3` 后 trial 行都存在，但 bind-only 的旧 mean fallback 随 run 一起丢失；min/median/max 原本也未被公开 trial 输出完整返回。显式 attach 的控制项保留。

修复在现有 prune 事务内，删除 run 前将完整 benchmark 小摘要保存到既有 trial 列；显式 attach 的摘要作为一个整体保留，避免拼接两次测量。公开输出增加已有 min/median/max；未增加 schema、采集职责或自动 verdict。回滚测试验证失败 prune 不留下半次快照。

独立审查进一步发现身份边界：重绑 trial 或复用已 prune 的 run ID，会把保留摘要误挂到另一 capture。现已拒绝不同 run 的重绑，并在分配新 run ID 时保留 trial/baseline 仍引用的 ID；同一现存 run 的绑定仍幂等。迁移路径是新建可链接的 child trial。旧库中早已丢失的历史身份不能靠本次升级重新证明。

冻结的合成历史还包含同一实现不同 workload、caller win 与外部 validation failed 冲突的事实。两种表示各 6 次共 12 个回答，11 个完全通过；一个 facts 回答错误地认为缺失字段已经存在。没有足够证据更换默认历史表示，但有明确理由修复已复现的保留损失。

## 当前开发取舍

完整盘点分两层：[41 个 MCP 接口的决定](../../profile_db/research/pfdb_development/interface_decisions.csv)记录状态、证据、反例、迁移与复审触发器；[92 项能力清单](../../profile_db/research/pfdb_development/capability_inventory.csv)进一步覆盖 27 个 query 的 48 个 view/selector/scope 模式、19 个生命周期模式、13 个实际支持的 render 组合与 12 个维护/通道模式。数字含不同模式，不是 92 个新工具，也不代表穷举全部参数组合。[生成器](../../profile_db/research/pfdb_development/inventory_capabilities.py)与完整性测试约束实际 schema 和支持范围。

- 保留 27 个 query 和 41 个 MCP 能力；没有证明底层分析可被等价删除，也没有证明新的统一目录更优。
- 修复可达性、合法空边图、trial 摘要保留；补充局部指标定义，修正 core/sparse/region/PMU 的过度说明。
- `compare` 明示兼容范围为 capture configuration / profile analysis，并返回 `workload_validation_checked=false`。这是已有 gate 的边界说明，不是新建 workload/validation 证明或接受机制。
- 完整契约存储、安全 PMU-only 绑定、通用 source/IR 索引和自动阶段拆分仍需各自证据；本轮未用不安全的 task-ID join 或自行拆分 fused latency 来填空。

没有任何底层能力达到删除门槛。候选按需分组没有完成预声明的十题交互目录对照；实际只完成单题说明预试，不能将其包装成分组方案胜出。静态列式在既定十题中压缩字节，但质量到顶且校准题存在反向成本例，因此保留默认 facts。原有手册中的过度因果说明、历史设计口径和旧工具计数作集中整理；不维护第二套 runtime 分析算法。

在线优化预试与独立确认均已完成；原输入 hash 全部不变。以下只将满足冻结 gate 的结果记为已测 fixture 收益，不将搜索最低数字、答题通过或确定性修复当作总体优化能力提高。

## 在线预试的冻结协议

选择局部指标语义与验证契约作为唯一附加信息条件 `semantic_contract`，与 A1 facts 基线比较；不更换工具目录或默认表示。两组具有同样的完整起点源码、四份 DSL 指南、固定 profiling 证据及外部执行器。模型不能读取仓库或调用工具，不会获得另一个实现的手写答案。因而本预试评估的是**给定证据包下的候选生成与反馈**，不是自主发现/调用 pfdb 的端到端效率。

后续反馈提供本会话的源码、source SHA、验证/计时状态与 benchmark，不提供新候选的 level-4、PMU、buffer hints 或生成物。审查者能读到的编译文件不等于 agent 可访问的证据。候选修改后再沿用起点 hint 的机制结论缺少新证明；既不能算模型忽略了可用的新 hint，也不能证明 pfdb 的自适应 profiling 工作流无效。本轮不据此判定 PMU、源码关联或显式 refresh 冗余。

冻结规模为四起点 × 三个重复编号 × 两条件，共 24 会话；每会话至多两个新候选。计划中的六候选是上限示例，本次在运行前选择较小预算并登记，未按结果延长。每次模型调用上限 180 秒，设备总墙钟上限 900 秒，其中队列等待上限 600 秒。失败消耗槽位并结束该会话；原诊断保留，不补跑到成功。

搜索使用无插桩 20 rounds / 5 warmup、同一 frozen golden、完整 median latency。起点先通过搜索输入和另一合法 fixture 的验证；候选不能修改 reference、comparator、imports、host、golden 或签名。AST guard 仅约束编辑范围，既不是操作系统沙箱，也不证明变换等价。模型会话之间独立，第二候选只可接续本会话并获得本会话反馈。

最终确认独立执行另一 fixture 的正确性验证和三个 100 rounds / 5 warmup 的新进程 strata。相同实现/输入的确认去重并记录共享，不将共享测量当成独立 agent 重复。外部 bootstrap 在三个固定 strata 内各自重采样 rounds，比较 pooled median；区间只条件于这些进程，不估计进程间或 agent 总体不确定性。原实现也必须有独立有效确认；缺失不得填 1×。候选未确认则退回已独立验证的起点，并单列回退。

匹配的编译产物先复制到确认私有目录再 replay；原始 build/capture 不作为 replay 目标。编译文件与 metadata 保留 hash 并在 strata 之间检查，避免运行器的缓存重建影响原工件。

设备纯排队时间无法从当前执行器可靠剥离，因此保留全部墙钟，不把估计等待从时间预算中扣除。提供方不暴露不可变 revision 或可复现 seed。上述限制、失败与逐对结果均须随结论一起保留。[优化执行器](../../profile_db/research/pfdb_development/run_optimization.py)、[确认执行器](../../profile_db/research/pfdb_development/confirm_optimization.py)

### 搜索预算的实际消耗

24 会话共 44 次模型尝试：30 次搜索验证通过、10 次失败、4 次主动停止。失败中有两个模型超时、一个明确的 `provider.no-route`，以及七个候选外部执行失败；后者包括已确认的数值失败和原因未定的编译阶段异常，不统一归咎 runtime。没有免费重试；提供方故障留在原 baseline 会话，已有 incumbent 保留，它也限制了条件效果的解释。

| 条件 | 会话 / 尝试 | 搜索通过 / 失败 / 停止 | 模型累计墙钟（秒） | 外部执行累计墙钟（秒） | session cost 合计 |
| --- | --- | --- | ---: | ---: | ---: |
| baseline | 12 / 22 | 14 / 6 / 2 | 1742.880 | 380.808 | 0.291109737 |
| semantic_contract | 12 / 22 | 16 / 4 / 2 | 1345.264 | 452.239 | 0.283965218 |

usage 为 24/24 会话取回的客户端 metadata，每个续接会话只计一次。上述 cost 仍是未经币种/结算核验的接口原值；墙钟包含等待与编译，不是纯 NPU 时间。提示和执行记录的存储/准备开销、独立确认与前置 capture 成本另列，不能用此表声称项目总费用或处理组显著更省。失败尝试不从分母移除。

### 独立确认与全部配对结果

11 个不同实现/输入组合全部通过保留 fixture，各自三个新的 100/5 raw strata 均成功，共 33 次计时；独立正确性/计时调用为 11 + 33，累计外部进程墙钟 856.405 秒。相同实现的测量共享，只计一次成本。所有 24 会话最终在已测 fixture 上通过验证，无缺失 latency，12 个会话保留了符合冻结 gate 的收益。baseline 为 5/12，处理组为 7/12；这些重复/共享测量不是 24 个独立硬件样本或总体成功率估计。

表中 latency 为 300 个原始 rounds 的 pooled median，单位 µs；speedup 使用同一起点的独立 baseline。重复编号不代表服务端受控 seed。`C4` 即 compressor_ratio4。

| 起点 | 重复 | baseline 最终 µs / speedup | semantic_contract 最终 µs / speedup |
| --- | ---: | ---: | ---: |
| SWA agent | 0 | 278.80 / 1.0000× | 278.80 / 1.0000× |
| SWA agent | 1 | 278.80 / 1.0000× | 278.80 / 1.0000× |
| SWA agent | 2 | 278.80 / 1.0000× | 250.40 / 1.1134× |
| SWA 手写 | 0 | 199.10 / 1.0000× | 199.10 / 1.0000× |
| SWA 手写 | 1 | 199.10 / 1.0000× | 199.10 / 1.0000×（回退） |
| SWA 手写 | 2 | 199.10 / 1.0000× | 199.10 / 1.0000× |
| C4 agent | 0 | 209.30 / 1.0000× | 201.20 / 1.0403× |
| C4 agent | 1 | 207.55 / 1.0084× | 207.55 / 1.0084× |
| C4 agent | 2 | 207.55 / 1.0084× | 207.55 / 1.0084× |
| C4 手写 | 0 | 220.90 / 1.1096× | 220.90 / 1.1096× |
| C4 手写 | 1 | 177.55 / 1.3805×（条件性扫描） | 220.90 / 1.1096× |
| C4 手写 | 2 | 232.50 / 1.0542× | 232.50 / 1.0542× |

SWA 手写 K128 候选在搜索中 194.7 µs、优于当时 195.7 µs，但确认得到 200.25 µs，相对独立起点 199.10 µs 为 0.9943×，区间为 [0.9925, 0.9958]，按原 gate 退回起点。这正是保留所有搜索结果并独立确认的价值；没有用其搜索优势报告成功。

四起点等权的配对 log-speedup 转回比值为 **0.99405494×**。只有三对有非零最终差异，其余九对最终使用相同实现（含回退）并共享测量；相同结果不是策略统计等价的证明。处理组在 SWA agent 和一个 C4 agent 会话得到不同改进，control 的一个 C4 手写会话得到更大的、带连续性前提的缩减扫描收益。不能据均值掩盖这个构成，也不能将较低失败次数或 7/12 对 5/12 写成显著提高。

**冻结 gate 验证的是这两个 fixture 上的正确性与时间，不是通用输入域等价。** C4 最大收益候选的保留 fixture 仍连续，无法裁定其 scan-8 前提是否属于算子契约。若用户要求任意合法 position/mapping，该候选的通用可接受性仍是 unknown；本轮没有将它替换进算子源码。另一个会话因同一前提未证明而停止，不应简单评为能力较差。这是需要权威输入/特化契约的具体证据，而非更多计数器的理由。

逐会话 source SHA、选定/最终 job、区间、验证、失败、共享关系、提示 digest、usage 与首次保留候选出现时间见[机器可读结果](../../profile_db/research/pfdb_development/optimization_summary.json)。12 个未观察到确认收益的会话保留删失/停止终点，包含提前失败，不只平均成功会话。首次出现时间是事后用独立确认标记的搜索执行墙钟，不是当时已经知道成功，也不含独立确认等待。

统计限制：区间只在固定三个进程 strata 内重采样 rounds，未估计进程间变化、轮间自相关、模型总体变化或多重比较校正；不据这些窄区间推断处理条件优劣。本轮使用相同 profile、超时与候选槽位，未设独立 token 上限，实际 usage 全部保留。完整固定材料预试及候选确认已完成；一般性质量/成本提升仍需新的独立 agent 会话与更广合法域/算子覆盖。

## 从失败区分信息缺口与变换能力

早期候选暴露了静态题分数未覆盖的错误：把 Right/L0B 的 buffer 压力等同 Vec/UB 压力，以及将删除恒等乘法的数学等价当成混合核同步等价。删除向量消费者改变了生成的 Acc→Vec 传递路径，候选出现大面积数值错误。另一个“32 blocks”方案实际新增的是逻辑 task，并在可用 capture 产生前失败。

这不是 runtime 指标错误的证据，也尚不足以判定编译器责任。已有 buffer hint 被错误使用、公开变换缺乏同步证明，分别属于解释与变换/验证环节；另一个候选缺少可定位的异常，应记诊断不足。详细来源、已确认事实和未决责任见[失败复核](../../profile_db/research/pfdb_development/online_failure_review.md)。没有修改上游或静默替换失败候选。

设计含义是保留 memory-space、生成 transfer 与同步证据的可达性，避免把这些事实压成“流水失败”结论。当前证据不支持为每次错误再增加一个顶层查询，也不证明单独补充时长定义即可保证合法优化。

随后出现另一个区别：compressor 的某些候选通过搜索验证并获得暂定收益，其解释却把正确的 40 个 work units 改说成 20 个，或把 32 个说成 256 个。源码和 shape 已在两组材料中，属于已有信息的关联/推导错误，不能直接归为“缺 profiler 指标”。SWA 相同 SHA 候选在不同搜索进程中跨过接受线，也不能据此分配处理组收益。逐候选的已知事实、解释错误和未决责任见[公开决策复核](../../profile_db/research/pfdb_development/online_decision_review.md)。

## 输入覆盖的边界

原始两实现均在分配条件前通过当前搜索输入及另一保留 fixture：compressor 使用 builder 已支持的 512 tokens / start=3；SWA 使用已支持的 causal regression / short window（17）fixture。共享 reference、comparator 和冻结数据；没有改宽容差或修改 golden 来制造合法反例。

这只扩大了两个明确输入的覆盖，**没有覆盖任意非连续 position、任意 physical mapping、多请求组合或全部组边界**。原始实现的通用等价仍未证明；新候选即使通过独立 fixture，也只能在已声明范围内记结果。RMSNorm/matmul 是指标语义 holdout，并未接受同样的在线优化评测，不能据它们宣称优化收益泛化。

## 静态研究的成本与复查入口

以下是主研究五批共 150 次尝试，不包含前置校准、启动失败、交互 smoke、设备补采或在线优化，不能当作本次项目的总成本。墙钟为各尝试之和，包含模型等待；并行执行时不等于用户实际经过时间。

| 批次 | 尝试 | 累计尝试墙钟（秒） | 提供方报告 cost 合计 |
| --- | ---: | ---: | ---: |
| D0 facts / columns | 60 | 575.682 | 0.062149079 |
| D1 信息增补 | 30 | 280.777 | 0.034617125 |
| E3 正向与 holdout | 42 | 394.816 | 0.037887022 |
| 实验记忆 | 12 | 239.807 | 0.034212450 |
| 探索性定义补充 | 6 | 45.820 | 0.004580975 |

这些批次的 session metadata 全部取回。`cost` 的币种/结算单位未独立核验，保留接口原数值，不标美元或推算账单。采集器只取会话级 cost/token metadata；不再叠加 message totals，不导出隐藏推理。静态逐题字段、引用覆盖、失败分类、提示 digest 与 usage 见[可公开结果摘要](../../profile_db/research/pfdb_development/study_summary.json)。实际 token/cache 字段保留提供方原口径，提示字节下降不等于账单下降。

执行器、隔离、重放与外部确认命令见[研究目录说明](../../profile_db/research/pfdb_development/README.md)。完整原始日志及大型 capture 位于忽略的实验目录；公开摘要保存来源与协议 SHA，不能替代原工件重分析。

## 发布检查与协议偏离

- 稳定工作树完整 CPU 回归：**378 passed、11 skipped**。跳过的真实 capture/环境依赖检查不冒充已验证；可用 SWA 与 holdout 的实际 parity 另见[升级审计](../../profile_db/research/pfdb_development/upgrade_audit.md)。保留既有 Pydantic `copy` warning。
- Ruff、版权头、公开文档边界/链接、72 页导航与差异空白检查通过。首个完整发布检查曾与 inventory 更新重叠：其他 377 项通过，快照一致性失败；保留原日志，修正清单后定向 3 项与稳定完整回归通过。
- 原版本到当前版本的变化、老库副本迁移、官方算法 parity、未知格式/版本拒绝与显式 refresh 已演练。没有为验收更换 runtime pin、发布软件包或提交 Git。
- 完成后再次核对：三个原数据库 hash、四个原算子源码、四个原始 replay build 全部不变；PyPTO/runtime revision 与 tracked binary diff 均保持 A0 值。原 compressor 的 35 个注册工件、独立工作库的十个 SWA 工件 hash 全部匹配。搜索与确认的 source/golden 输入检查均通过。
- 离线评分使用冻结的精确值/类型与引用 ID/覆盖规则；不是自由文本 entailment 证明。候选复核独立执行，但审查者可见条件与结果，**未完成预声明的自由解释盲评**；本轮不据其构造处理组质量分数或显著性结论。
- 没有完整十题交互分组目录对照，也没有在线记忆、逐组件消融或新算子在线优化收益确认。相应方案未被提升为默认，底层能力保留；这是明确未评估范围，而非删除证据。

后续是否需要额外实验，按[完整计划审计](../../profile_db/research/pfdb_development/completion_audit.md)和逐接口复审触发器判断。任何未来默认改变、删除或一般性 agent 收益主张，仍须满足对应的质量/成本与独立确认门槛。

## 下一轮最值得验证的三件事

| 优先项 | 本轮具体触发证据 | 最小下一步，而非预先扩张 schema | 放行条件 |
| --- | --- | --- | --- |
| 输入特化契约与 validation 绑定 | 同一扫描热点，一个会话依赖连续 position 得到大收益，另一个因前提未证明而停止；保留 fixture 也连续 | 先在外部证据包明确“允许依赖的模式”与“只在采样中出现的模式”，加入经 spec 确认的反例；对照 agent 是否正确条件化接受 | 不将 initializer 的 arange 当作通用契约；缺失信息仍为 unknown；不改 reference/comparator |
| work/grid/row/core 与内存池关系 | 40/20/256 units 混淆、Right 与 Vec 混淆；关键原始值已经可见 | 复用三类出错题做短关系事实原型，与完整旧 facts 比较；关系必须带 source/build 锚点和计算依据 | 纠正具体错误且不引入错误计数；只有通过后才考虑目标局部索引，不先建通用 IR 引擎 |
| 新候选证据的导入与失效边界 | 本轮仅回传 bench；旧 hint 被沿用；新编译文件虽已产生却不在模型可见反馈中 | 外部 harness 导出明确绑定候选 SHA 的现有 hints/异常/生成物摘要，按相同权限对照有/无这些事实 | 不把新 build 附到旧 run 后假装同一证据；无法绑定时拒绝自动关联；pfdb 仍不采集或运行 |

这些优先项来自可复核的错误与成功候选，不等于相应新 API 已获收益证明。工具目录压缩、通用列式、独立 PMU join、在线记忆继续保留为分开的候选；新增工具必须有现有组合不能回答的问题。低频能力已有正向 fixture 而缺少真实优化收益，适合保留按需可达，不宜因两例未使用而删除。

2026-10-22 的后续实验与新增边界见[输入契约、候选身份与工作关系研究](pfdb-followup-results.md)。新实验独立登记，未改写本轮冻结 packet、响应或历史性能数字；SWA 的完整分配关联明确了两版都有两个 Right 分配，变化发生在外层 K chunk 与它们的对应关系。
