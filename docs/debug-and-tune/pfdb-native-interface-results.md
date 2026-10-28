# pfdb 0.7–0.8：诊断读取、执行规模与原生工具验证

日期：2026-10-22。承接[五道新算子研究](pfdb-new-operator-study.md)，本轮实现两个已有接口的增量能力，并把它们交给 OpenCode Go 的 `opencode-go/deepseek-v4.1-flash` 实际查询。**功能闭环已经验证；尚未证明完整算子优化收益，也没有得出可以删除哪些工具的结论。**

以下第 1–8 节记录 0.7 阶段。随后增加的 24 会话目录对照、12 会话语义对照和 0.8 实现见第 9–11 节；分阶段结果不混合评分。

## 1. 实现依据与边界

上一轮五题包含 RMS、lookup、HC、conv 和 MTP。暴露的三个直接问题是：编译失败时缺少可直接读取的诊断正文；SPMD 的执行规模需要查找 task/rows 才能拼出；真实候选的身份与范围仍需要人工组装 manifest。它们优先影响 agent 能否取得、关联事实，并不等于新建一套瓶颈分析算法。

本轮改动如下：

| 问题 | 实现 | 没有据此声称什么 |
| --- | --- | --- |
| 有诊断锚点，未必能直接读取正文 | 现有 `list_trials` 增加 opt-in `view=diagnostics` | 不自动判断编译错误原因，不执行日志指令 |
| 执行规模的基础事实分散 | 现有 `overview` 增加声明 block 计数和按 engine 的物理行/核心计数 | 不把计数叫作利用率、执行波数或最优任务粒度 |
| 后续候选需要人工组装身份工件 | 外部研究执行器在采集进程退出、日志关闭后自动导出 manifest | 数据库不采集，不自动认证正确性，不伪造构建身份 |

包版本为 **0.7.0**，数据库仍为 schema **9**，MCP schema 升为 **6**，工具仍是 **41 个**。运行时分析算法、adapter `pfdb/0.4.0` 和已有 analysis snapshot 不因本次改动重算；详见[复用与升级台账](pfdb-runtime-integration.md)。PyPTO/runtime、手写算子、原数据库和原始 capture 均未修改。

## 2. 诊断读取的具体契约

```bash
pfdb trial list --trial-id 4 --view diagnostics \
  --source-sha256 <expected-source-sha256> --budget 32768
```

必须提供 trial 和预期 source SHA；scope SHA 可额外限制范围。默认 summary 的行为保持不变，不读取正文。新 view 只返回 checked candidate 和注册诊断，避免同时重复整份 artifact 列表与 benchmark。

读取器只接受 manifest 注册的文件，核对整个文件的 hash，再返回同次读取保留的末尾至多 **16 KiB 源字节**。正文带总大小、偏移、截断状态和 UTF-8 替换解码标记。路径越界、非普通文件、文件改变、source/build 陈旧或 SHA 不匹配时不返回正文；返回前再次检查 source/build。内容仍是生产方声明的证据，不是 pfdb 的解释。

预算截断和内容截断分别报告。大文件即使只返回末尾，也需要完整读取以验证 SHA；输出预算不是内部 I/O 上限。需要更早片段时仍可使用注册文件的原生读取能力，当前接口不增加任意文件读取或分页系统。完整契约见[候选证据](pfdb-candidate-evidence.md)。

## 3. 执行规模摘要的含义

`overview` 的 `METRIC` 增加 `declared_blocks_sum` 和 `block_counts_missing`。NULL/负数的未知 block 不参与求和，缺失数量单独报告。每个记录中出现的 engine 返回一个 `EXECUTION`：

- `physical_rows`：物理记录行数；
- `distinct_cores`：这些记录实际出现的不同 core 数；
- `tasks_with_rows`：这些行关联的不同 task 数；
- `max_rows_per_core`：单个 core 的最大记录行数。

mixed task 按**物理行记录的 engine** 分组。没有物理行时返回 unavailable，不把未采集解释为零执行。拓扑容量仍来自原有 `RESOURCE`；以上都是整次 capture 的计数，不提供时间加权利用率，也不能把行数直接等同于逻辑工作单元数。

真实 conv 工件的结果为：1 个 task、声明 6 blocks、6 条 AIV 记录、6 个实际出现的 AIV core、每 core 最多 1 条记录；同一响应中的拓扑为 20 AIC / **40 AIV**。这些足以描述此次执行规模，不足以证明应该再细分任务。

## 4. 外部生产方的自动交接

[`export_new_candidate.py`](../../profile_db/research/pfdb_development/export_new_candidate.py) 被外部 `run_new_operators.acquire` 调用。它在运行前保留 candidate、题目 reference 和实际 comparator 的快照及 hash，运行后拒绝 source/reference/comparator 的变更或 collection 报告的错配。

成功、可绑定 fixture 的结果锚定实际 collection 报告、reference、comparator、fixture 文件清单和验证范围；输入 golden 文件改变或缺失时，不声明该 fixture 已通过。编译或进程失败保留诊断并使用 validation `not_run`，不将任何执行失败都归类为数值失败。bench 只有实际日志可解析时才登记；profile 只在 records/deps/name_map 完整时交接，缺失仍显式保留。

conv 记录 `evolve_inout_after_initial_validation;fixed_qkv`，说明计时期间状态会演进，并声明没有逐轮验证每个 timed output。数值 PASS、业务合法域、并发安全和性能收益仍是不同问题。

这个增量没有把完整构建/工具链来源自动认证：`build_id` 保持 unavailable，并明确记录限制；上一轮独立回放中已有构建锚点的候选仍可照常查询。生成 manifest 不会隐式创建数据库或 ingest；外部调用者通过已有 `trial register` / `attach --trial` / `bind` 接入。原始 capture 不被回写。

实际已有工件回放补充：lookup 的 profile manifest 成功导入并绑定真实 run，conv 的 bench-only manifest 成功导入；二者仍明确 build unavailable，并拒绝错误 source 的诊断读取。旧 RMS 失败 process 记录没有 source SHA，自动交接正确拒绝，未根据当前文件倒填历史身份。这个检查不重新执行数值验证；它验证既有报告及 frozen fixture hash 的交接。

回放还发现并修复了一个 **pfdb 自身的状态聚合错误**：原来只要任意注册文件是空文件，整个 modality 就返回 `empty`。真实 stdout 为 12,608/5,636 字节、stderr 为 0 字节的两例都被误标为空。现改为所有文件完整、至少一个非空时 `available`，所有文件为空才 `empty`；任意缺失/变更仍为 `unavailable`，逐文件状态保留。修复前两条 handoff 回归测试稳定失败，修复后已有数据库只读查询正确，无需改原工件或重采。这说明状态信息本身出错，也会使 agent 错过已有证据。

## 5. 原生查询实验与 setup 失败

[`probe_native_interface.py`](../../profile_db/research/pfdb_development/probe_native_interface.py) 固定三题、每题两次新会话，最多 12 步/会话、600 秒超时，不重试模型失败。期望答案在调用前由只读数据库导出，对模型隐藏。模型只可调用只读 pfdb，不可读手写代码、运行命令、改数据、启动设备或代理。使用数据库副本，副本留在原 DB 的同级目录，保留相对工件定位基准。

初始校准不能算作 agent 能力样本：

| 阶段 | 模型会话 | 观察与处理 |
| --- | ---: | --- |
| 三个 transport smoke | 3 | 前两次未取得 pfdb 正文；显式 direct-tools 的一次成功读取 RMS 诊断 |
| native-queries-v1 / v2 / v3 | 18 | 后端确实列出 41 tools，但没有实际 pfdb 调用；输出包含未执行的工具调用文本，不能当作查询成功 |
| native-queries-v4 / v5 | 0 | 新预检分别拒绝未就绪的 agent 和 pending MCP；在模型调用前停止 |
| native-queries-v6 | 6 | 先验证 agent/连接，再在同一已初始化 location 查询；取得实际 pfdb 调用 |

这里同时调整过权限与服务初始化方式，**没有做足以单独归因每个客户端设置的控制实验**。因此不能将 18 次失败归因为模型不懂 pfdb，也不能断言某一个 permission glob 就是全部原因。已确认 OpenCode 默认使用 Code Mode，需要 `execute`；本轮只读实验选择 `codemode=false`。客户端确实曾依次返回空/部分 agent catalog 和 pending MCP，故执行器现在进行有上限的就绪检查，并在失败时保留诊断、拒绝模型调用。

校准共保留 21 个模型会话，正式原生验证 6 个，本轮合计 **27 个**；未采集新的设备数据。原始 stdout/stderr、配置、预注册、逐次结果和后端 trace 留在忽略的 `build_output/pfdb_development/`。它们没有被“补跑成功”后覆盖。

## 6. 正式六会话：检索成功不等于回答完全忠实

| 题目 | 重复 0 | 重复 1 | 审计结论 |
| --- | --- | --- | --- |
| RMS，读预期 source 的编译正文 | 严格评分通过，3 次工具调用 | 严格评分通过，1 次工具调用 | 都取得 `truediv` 的 `fp32 and index` 错误；没有把文件存在当作正文 |
| conv，读执行规模并说明证明边界 | 严格评分未通过，2 次调用 | 严格评分未通过，2 次调用 | 两次都取对四个目标计数并否认仅凭计数证明利用率不足；仍存在下述问题 |
| 错误预期 source，不得借用正文 | 严格评分通过，1 次调用 | 严格评分通过，2 次调用 | 都报告 mismatch，未取得该错误身份的诊断内容 |

预注册的严格评分为 **4/6**，保留这个数字，不为提高分数修改规则。两次 execution 的失败原因不同：

1. 重复 0 先调用不存在的 `pfdb_pfdb_pfdb_rows`，还把 engine 字符串 `aiv` 用作 task_id；实际 `overview` 查询成功。最终四个计数正确，却在解释中写了 **64 AIV cores**。工具响应明明是 **40**，这是额外事实的无依据补全。主要评分因工具错误已经失败，但它没有直接检查解释中的拓扑容量，说明只检查少数字段会漏掉实质错误。
2. 重复 1 没有工具错误，容量解释也正确，但把四个计数放进 `answer.aiv`，而 scorer 要求平铺字段。问题文字没有明确约束平铺，故这是评测格式的不充分规定，不能算成数据库检索失败或性能理解失败。

六会话共有 **11 次工具调用尝试**，其中 10 次执行到后端、1 次工具名错误；客户端记录的成功响应正文合计 **10,957 字节**。这不是模型全部输入 token，也不包含完整工具 schema 的成本。没有发生响应预算截断。数据库原件与查询副本的文件 SHA 均未改变。

27 个会话的 aggregate usage 元数据全部取得：input 133,522、output 5,426、reasoning 数量 26,978，API 原生 cost 合计约 0.03919；其中正式六会话 input 99,849、output 2,973、reasoning 数量 2,057，cost 约 0.01822。未取得货币或 Go 订阅结算口径，不能作为用户实际费用。reasoning 仅统计元数据数量，没有读取其正文。完整工具 schema 成本没有单独计量，不能用 10,957 字节响应总量代替全部上下文成本。

后续评测应分别记录：工具是否真实执行、目标事实是否正确、额外解释是否有来源、期望身份是否保留、接口/格式失败能否恢复。不能将“最终 JSON 通过”作为唯一质量标准。

## 7. 对 pfdb 下一步取舍的约束

已经有依据保留的方向：

- 让当前候选的 source/scope/缺失状态与诊断片段能直接取得；不要求每次失败先造一个 profile run。
- 默认摘要可以提前给执行规模，但必须保留时间和容量的证明边界。
- 外部执行器应交接真实工件及状态演进，不能要求数据库推测候选关系。
- 评测需要执行工具的 agent；静态 packet 不能覆盖客户端初始化、工具命名、selector 和恢复行为。

仍不足以直接实施的接口改造：

- 不能因一个重复前缀错误就改名全部生产 MCP 工具。先在实验 server 做“相同能力、仅改变命名”的对照，保留现有调用者兼容性。
- 不能因容量出现在 `RESOURCE` 而被 agent 忽略，就宣告这些信息无用；可以研究与执行计数共同呈现、逐字段证据引用，但要固定事实集，避免把新增事实算作格式收益。
- 原生文件读取与 source-guarded pfdb 内容读取的优化效用对照尚未完成；本轮只证明后者能工作，不证明它优于前者。
- 仍没有新的 PMU/memory/scheduler 正向优化与消融证据，不删除这些能力，也不新增自动瓶颈判断。

这些不确定性约束下一轮实验，而非通过重复增加默认字段来填补。完整自主开发的性能对照沿用[新算子研究的 P1](pfdb-new-operator-study.md)，应与本轮接口验收分开报告。

## 8. 验证和复查入口

- [可移植实验摘要](../../profile_db/research/pfdb_development/native_interface_summary.json)：逐次严格评分、后端调用数、原始输出 hash、公开响应人工审计与 aggregate usage。
- 生产接口：[`candidate.py`](../../profile_db/src/profile_db/lifecycle/candidate.py)、[`handlers_z0.py`](../../profile_db/src/profile_db/query/handlers_z0.py)。
- 诊断边界测试：[`test_candidate_diagnostics.py`](../../profile_db/tests/test_candidate_diagnostics.py)；覆盖变更、越界、读取期间改变、缺失、二进制、空文件、预算恢复与 API/CLI/只读 stdio MCP 一致性。
- 执行计数测试：[`test_execution_overview.py`](../../profile_db/tests/test_execution_overview.py)；覆盖 mixed task、未知 block/engine、未执行 task、无物理行及 golden overview。
- 外部交接测试：[`test_candidate_handoff.py`](../../profile_db/research/pfdb_development/test_candidate_handoff.py)；不调用设备，验证失败不会伪装为数值失败、报告/fixture 错配及完整/不完整 profile 交接。
- 原生 runner 与评分测试：[`test_native_interface.py`](../../profile_db/research/pfdb_development/test_native_interface.py)；客户端预检失败不进入模型请求。
- [当前能力清单](../../profile_db/research/pfdb_development/capability_inventory.csv)增至 **95** 个 view/selector/导出组合；41 tools 没有变化，历史 A0 清单没有重写。

修复后的 CPU 全量回归已完成 **463 passed / 11 skipped**；随后候选/诊断、外部 profile 元数据及客户端预检的针对性检查为 **55 passed**（与全量测试有重叠，不相加），Ruff 已通过。没有进行新的 NPU 性能测量，以上六个会话不生成算子，也不能替代下一轮性能对照。文档站点完整构建和跳过测试对应的运行环境仍不在本轮已验证范围内。


## 9. 目录对照：更少 schema，不等于更少查询或更高准确率

后续使用同一批真实工件、四个固定问题（诊断正文、执行规模、错误 source、PMU 缺失），做 **2 个目录 × 4 题 × 3 次独立重复 = 24 个新会话**。模型仍为 OpenCode Go 的 DeepSeek Flash。原生 `codemode=false`，每会话 12 步、600 秒、执行器不重试，配对顺序交替；脚本见 [`probe_catalog.py`](../../profile_db/research/pfdb_development/probe_catalog.py)。

对照是当时 0.7 的完整 41 工具目录；处理组仅暴露 `overview`、`inventory`、`list_trials` 三个工具。三者保留全部回答所需的事实和原有 schema/返回值，其他能力在处理组不可访问。因此这是**任务限定的目录裁剪**，不是把全部能力无损压成三种接口，也不能据此判断完整优化工作流中的工具冗余。实验 server 的过滤不进入生产。

两个条件使用相同题目、明确的平铺答案字段、隐藏的预期答案及 source guard；新增容量字段被显式评分，避免只核对四个计数、漏掉解释中错误容量的问题。这个新协议不能与前述六会话的不同题目直接作准确率对照。

| 观察量 | 完整目录 | 三工具目录 |
| --- | ---: | ---: |
| 实际模型会话 / 正常结束 | 12 / 12 | 12 / 12 |
| 严格通过 | 10/12 | 9/12 |
| 目标字段正确 | 11/12 | 9/12 |
| 执行规模题严格通过 | 3/3 | 0/3 |
| 工具调用 / 后端实际调用 | 27 / 27 | 32 / 32 |
| 响应正文总字节 | 25,297 | 58,740 |
| 序列化 MCP 工具对象总字节 | 39,689 | 3,245 |
| API aggregate input tokens | 266,636 | 58,207 |
| API aggregate output tokens | 6,546 | 6,346 |
| API aggregate reasoning 数量 | 4,528 | 4,899 |

工具对象体积减少约 **91.8%**，记录到的 input tokens 减少约 **78.2%**；12 个配对的 input 比率中位数为 **0.213**。响应正文反而增至约 2.32 倍，查询次数也更多。工具对象字节不是模型全部输入；token 还包括问题、重复上下文、前序结果和模型自己的工具选择。

严格配对结果是：两组都通过 7 对、仅完整组通过 3 对、仅三工具组通过 2 对、两组都失败 0 对。这个样本不足以证明三工具目录非劣，也不支持“完整组一般更准确”。

具体失败比总分更有解释力：

- 完整组一次诊断回答把正确 source SHA 多复制了两个字符，却声称逐字复制；**实际查询参数里的 SHA 是正确的，正文也取对了**。这是回答转录错误，不是访问了错误候选。系统关联应使用调用方计算的 SHA 与原始机器事实，不能依赖模型重新抄写的摘要。
- 完整组一次 PMU 回答目标字段全对，但多发起了一个缺少 family/name 的 `tasks` 查询，严格评分因工具错误失败。
- 三工具组的三次执行题都读到了 `RESOURCE cores=40 engine=aiv`，却将要求的 `topology_aiv_cores` 填为 unknown/null。其解释明确区分了“resource capacity”和自己理解的“topology”。这也暴露题目中的未限定术语：预期答案是 capture 声明的数量，不是独立认证的芯片硬件上限。
- 完整组使用 `density` 的 `total_cores=40` 帮助解释该数量。这与 RESOURCE 是**同源的元数据呈现**，不是第二次独立硬件测量；它在某次查询中具有解释价值，不能仅按数值重复就删除。

全部 24 个会话的 aggregate usage 已取得。API 原生 cost 合计为完整组约 0.04701、三工具组约 0.01563，货币及 Go 订阅结算口径仍未知。一个完整组会话在 pfdb 查询已返回后等待至约 466 秒才完成，没有重试或超时；墙钟最大值不能解释为数据库性能。两组墙钟中位数约 26.9/16.7 秒，也不作为 pfdb 延迟基准。

可复查数据见[目录对照摘要](../../profile_db/research/pfdb_development/catalog_summary.json)，其中记录每个失败字段、调用数、原始响应 hash 与用量。独立检查全部诊断调用，没有发现改用另一 source guard 读取正文的行为。

## 10. 把问题歧义与事实缺少范围标签分开

目录实验后，增加一个 **2×2、每格 3 次、共 12 个新会话** 的原生查询试验，固定三工具目录和所有数值：

1. 问题：沿用原问题，或明确规定该数量表示 capture 声明的 AIV 数量；
2. 返回事实：沿用原 RESOURCE，或仅在它上面增加 `scope="capture_declared_topology"`。

后者来自现有 `run.core_types` 元数据计数的既有定义。它没有新增硬件观测或容量算法。问题定义与事实 scope 都增加了**语义信息**，不能称为纯格式压缩。实现与预注册见 [`probe_resource_scope.py`](../../profile_db/research/pfdb_development/probe_resource_scope.py)；12 次均为新会话，不复用第 9 节的三次失败作为对照。

| 问题 / 返回事实 | 严格通过 | 工具调用总数 | 响应正文 B | 记录 input tokens |
| --- | ---: | ---: | ---: | ---: |
| 原问题 / 原事实 | 2/3 | 9 | 12,558 | 15,378 |
| 原问题 / 加 scope | 3/3 | 6 | 8,116 | 11,438 |
| 明确定义 / 原事实 | 3/3 | 6 | 7,788 | 9,963 |
| 明确定义 / 加 scope | 3/3 | 5 | 6,066 | 8,835 |

新一轮原问题/原事实并没有重现上一轮的 0/3，而是 2/3，说明重复波动和样本限制必须保留。唯一失败仍是容量字段的保守 unknown；没有新造错误数值。实验支持继续澄清来源和范围，**不足以声称 scope 标签已经带来一般准确率提升，更不能声称带来算子加速**。问题描述本身也能解决歧义，不能把所有收益归给数据库。

[语义试验摘要](../../profile_db/research/pfdb_development/resource_scope_summary.json)保存逐次结果及源输出 hash。所有条件中工具 schema 一致，只有指定问题定义或 RESOURCE 的 scope 注解变化。

## 11. 据此落实到 0.8 的最小改动

本轮没有将默认接口缩成三个，也没有删除低频分析能力。实施了两项边界明确的改动：

- `overview.RESOURCE` 增加 `scope="capture_declared_topology"`。数值、排序、证据状态和查询方式保持相同；该数量表示 capture 的声明，不独立认证硬件峰值或正在执行的核心数。观察到的核心数仍在 EXECUTION 中。
- 只读 MCP 仅广告 **33 个数据库读工具**；已有 `--writable` 启用全部 **41 个**。被省略的八项是 attach、refresh、baseline_add、register_trial、bind_trial、attach_bench、set_verdict、note，它们原本在只读连接上始终被拒绝。这是按实际权限清理目录，没有移除任何合法读操作。直接调用的写入保护仍执行；可写会话也已验证能正常 register。

全部能力仍为 41，当前能力清单仍为 95 个组合。只读目录的序列化工具对象为 **35,195 B**，比完整 41 项减少约 **11.3%**。这个逻辑上不可用的部分可以清理；它与第 9 节的任务限定三工具方案不同，后者没有进入生产。没有用这些字节变化代替模型准确率或性能收益验证。

版本为 **pfdb 0.8.0 / MCP schema 7 / DB schema 9**，candidate manifest 仍为 v1。客户端升级后重新连接以刷新工具目录；已有 runtime 分析无需仅为此次升级 refresh。PyPTO/runtime 和原始工件不变。

验证包括 33/41 真实 stdio 目录、所有隐藏写操作的保护、可写注册成功、只读 DB 字节不变、API/CLI/MCP 和 golden facts 一致性。一个新 OpenCode 集成 smoke 已实际发现 33 工具，正确读取声明 40 核、观测 6 核、scope 标签与 schema 7，并正确报告 register_trial 未提供；这只是集成检查，不是性能对照。

两项试验均冻结了执行器源码、包源码 hash、原始 DB hash、问题与顺序；运行期间 69 个包源码文件和 DB 保持一致。为支持升级后的复查，另存了完整包源码：其中三个已升级文件在研究完成后，仅在归档目录中逆转小改动，再逐字节验证 SHA 与预注册记录相符。归档是**事后重建并经预注册 hash 验证**，不是声称事前已经复制了这些文件。当前工作树和原始采集没有被回退或改写。

剩余边界：两批 36 会话研究检索、解释与成本，不生成内核；加上前一阶段和新的集成 smoke，也不能替代具有独立数值验证及性能确认的自主优化实验。后续若研究渐进发现或客户端 Code Mode，应保留完整分析能力，用新的匹配对照验证，不能把所有接口合并后再用绿色单元测试证明 agent 会优化。


0.8 最终检查：**471 passed / 11 skipped**，Ruff、文件头和文档链接检查通过；仍有一条既有的 Pydantic `copy` 字段遮蔽提示。生产 overview 与六个实际 scoped 原型响应逐字节一致。293 个保护文件、原数据库和 PyPTO/runtime 的提交及 tracked diff 未变。汇总证据见[0.8 验证记录](../../profile_db/research/pfdb_development/pfdb_080_verification.json)。跳过测试对应的环境、文档站点完整构建和端到端优化收益不在此次已验证范围内。

外部执行器另补充了超时后的 owned-session 取消路径，避免只终止 CLI 而遗留共享服务中的请求；该路径用模拟测试验证，没有真实会话达到 600 秒超时。追加针对性测试为 12 passed，与全量测试有重叠，不相加。
