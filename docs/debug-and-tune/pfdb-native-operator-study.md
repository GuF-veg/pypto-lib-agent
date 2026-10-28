# pfdb 0.8：原生算子闭环的可行性研究

日期：2026-10-22。承接[新算子研究](pfdb-new-operator-study.md)与[原生接口研究](pfdb-native-interface-results.md)。本轮把评价单位从“回答一个 profiling 问题”推进到“读取证据、提出源码修改、外部验证修改”的轨迹。

本轮得到更完整的正反例：Go 的 RMS 与卷积候选在独立确认中快于手写控制，HC 仍较慢；MTP 则暴露了编译、资源约束和数值错误，且发现一个可独立复现的编译器双重舍入缺陷。已经据确定性证据落实 pfdb 0.8.1 的四处语义修正，没有新增顶层工具。v3 受服务中断影响，v4 随后受共享盘写满影响；原结果和失败均保留，不能用它们估计诊断入口的总体效果，也不能据工具调用频率删除分析能力。

## 研究问题与对照范围

问题是：在已有源码、编译失败记录与 profile 的条件下，通过 pfdb 读取带身份校验的诊断正文，能否帮助 OpenCode agent 得到更好的候选？两组都保留相同的 pfdb 只读分析工具，因此比较的是**诊断正文入口**，不是“有 pfdb”对“完全没有 pfdb”。

| 因素 | 固定条件 |
| --- | --- |
| Agent | OpenCode 2.1.2；opencode-go/deepseek-v4.1-flash，订阅通道 |
| 题目 | RMSNorm、MTP projection、HC post、带状态更新的 KDA convolution decode |
| 重复 | 每题每组 2 条轨迹 |
| 修改预算 | 每条最多 4 次；失败、格式错误、越界修改都消耗预算；允许主动停止 |
| 会话 | 每次提案是新会话；控制器提供当前完整源码和此前轨迹历史 |
| 工具预算 | v3/v4 每会话最多 8 steps，要求至多两批查询；33 个只读工具、不开 Code Mode；v2 的 4-step 失败另列 |
| 可修改范围 | 仅该题 EXAM 函数体；签名、decorator、imports、全局常量、host、reference 均冻结 |
| 搜索验证 | 同一个冻结 golden；a2a3，npu-run 分配设备，逻辑设备 0 |
| 搜索计时 | 完整未开启 profile 的 device effective latency，20 轮、warmup 5 |
| Profile | 仅数值 PASS 后另跑 level 4；不以 profile makespan 代替未开启 profile 的延迟 |
| 人工介入边界 | 人工不修复 agent 算子，也不提供手写优化实现 |
| 身份限制 | source SHA 来自控制器实际文件；build identity 保持 unavailable |
| 确认 | 对保留且经安全审查的源码，验证额外 fixture，并作三个独立 100/5 进程确认 |

两种处理如下：

- native_file：通过 OpenCode 原生 read 读取隔离 workspace 中的诊断文件；实验 MCP 的 list_trials 只提供 summary，直接请求 diagnostics 也被拒绝。
- pfdb_diagnostic：通过 list_trials 的 diagnostics view、trial_id、source_sha256 读取；不允许原生文件读取。

两组提供的诊断文本逐字节一致，原始 stdout/stderr 均不改动。外部控制器连接已注册诊断，保留末尾至多 16 KiB，经 UTF-8 replacement 解码并约束最终字节预算，生成新工件。记录原始总大小、原始内容 hash、各原始日志 hash、新工件大小与 hash。派生文件同时是 pfdb 注册锚点和文件组副本；两者内容不一致就不开始模型调用。

这不证明原始日志尾部包含全部关键线索。若根因在被截去的前文，两组同样受限；后续可以另行检验诊断分页是否有价值。

## 为什么保留这四种结构

起点使用上一轮 Go 自己生成的候选，避免把手写解法交给模型：

| 题目 | 重新验证的起点 | 能观察的不同失败类型 |
| --- | --- | --- |
| RMS | 编译失败 | DSL 类型与 lowering 边界；能否把真实错误转换成正确修改 |
| MTP | 编译失败 | 混合量化/矩阵乘与 padding；框架限制是否被误当成调优问题 |
| HC | 数值 PASS，已有 profile | 合法分块、任务数量和融合；数值通过与并发写安全是否混淆 |
| Conv | 数值 PASS，已有 profile | 串行单 task 与并行切分；InOut 状态语义是否在优化中保留 |

这不是题库随机抽样，也不是总体成功率估计。前两题区分“信息没取得”“读到错误但修错”“触碰框架限制”；后两题区分“观察正确”“提出合法优化”“降低完整延迟”。

lookup 已有成功例子，本轮优先检验诊断和状态/写入范围边界，没有继续增加其重复。不能因此判断 lookup 相关能力冗余。

## 执行与证据交接

run_native_operators.py 是外部研究执行器，不是 pfdb 功能。每次提案的顺序是：

1. 使用已有工件建立私有数据库、trial 和候选 manifest；有 profile 时再导入和绑定 run。
2. 关闭写连接，等待隔离 OpenCode agent 和只读 MCP 实际 ready。
3. 模型自主选择查询，返回函数体、假设、引用证据、缺失证据和下一实验。
4. 控制器检查修改范围，保存候选，通过外部采集器编译和验证。
5. 进程退出、日志关闭后 producer 导出 manifest；控制器派生下一份证据 bundle。
6. 数值 PASS 才进行 profile；原始文件和此前候选不回写，不混接不同 source 的 bench/profile。
7. 下一轮提供新 source 的 SHA、trial/run、状态与轨迹历史，重新建立会话。

范围检查复用 AST guard，不靠模型回显身份关联结果。编译失败没有 run，仍能注册 trial 和读诊断。summary 只显示身份、范围和可用性，正文须显式请求。运行时算法和口径继续来自 runtime，没有另造关键路径或利用率算法。

顺序预先固定：第 1 次重复 file 在前，第 2 次重复 pfdb 在前，最多 4 条轨迹并发。这只是平衡提交顺序，不能消除服务端负载、模型随机性和设备状态影响；没有 immutable provider model revision。

## 校准失败单列

第一次批次 native-operator-v1 因**研究执行器配置错误**停止。文件组权限资源用了绝对路径，而 OpenCode 对 workspace 内文件按 location-relative 路径匹配；读取事件参数名应是 path，不是控制器最初假设的 filePath。日志显示 Permission denied: read，不能当成“拿到相同证据之后仍然失败”。

这批会话和已生成候选保留，不计正式对照。修正后另设 native-read-permission-calibration，用无算子内容的标记文件确认实际事件为 read:completed，再启动 native-operator-v2。权限解释参照 [OpenCode 官方文档](https://opencode.ai/v2/docs/permissions#actions)，并以本机真实事件验证。

v2 协议在首次调用前冻结，包含脚本和 pfdb package hash 及归档。但 v2 随后也停止：4-step 上限会触发客户端的强制总结指令，模型多次返回“Maximum steps ...”的自然语言总结而非所要求的 JSON 函数体。该批次不能回答正式有效性问题。

v1 实际启动 7 个模型会话，3 个完成控制器处理，1 个候选进入编译；v2 启动 12 个会话，8 个完成控制器处理，3 个候选进入编译；两批均无数值 PASS。未完成会话由执行器所有者显式中断并关闭对应 MCP，不算作已交付的失败候选。配置校准额外使用 1 个只读 marker 会话。上述 20 个会话的 session usage metadata 均取得，不能只报告“成功校准用了 1 次”而隐藏前面的成本。

两批原协议及归档保持原样。native-budget-calibration 已完成两条 RMS 短轨迹，每条 2 个提案、8 steps，要求最多两批针对性查询。4 次均提交了合法 JSON、通过源码 guard 并进入编译；数值 PASS 为 0。客户端耗时分别为 file 组 113.053/549.036 秒、pfdb 组 81.840/93.689 秒。549 秒会话最终正常返回，无超时或补跑；样本不足以把这个长尾归因于读取入口。四个 session 的 usage metadata 均取得。校准没有证明编译成功或优化收益，但证明了两组都能走通“真实工具读取→函数体提交→外部编译”流程。

随后启动 v3。它从相同 bootstrap 源码出发，重新冻结全部实验脚本和 pfdb Python 源码，不复用校准提案。输入中的工作负载、golden、reference、源文件校验继续生效。

预算校准还修正一处评价不对称：越界 read 被客户端在读取前拒绝时，只记工具错误，不整份丢弃随后合法的 JSON 候选；若实际读取了不授权内容，则仍拒绝。这样与 pfdb 参数错误后允许恢复的规则一致。v1/v2 的原评分不回写。

## 评价规则与限制

读取入口成功不等于修复成功，修复成功不等于提速。分别统计：

- 会话是否正常完成、实际用了哪些工具、是否取得诊断正文。
- 参数错误、预算截断、source guard 拒绝、其他工具错误和格式错误。
- AST guard 接受、编译成功与数值 PASS。
- PASS 候选是否存在跨 block 重叠写、尾块越界、状态更新遗漏或仅适用于搜索形状的假设。
- 每条轨迹保留候选的搜索与独立确认延迟；同 source 去重，共享候选不能算作多个独立发现。
- 额外 fixture 是已有测试、探索性挑战还是业务输入域 authority；失败不自动证明所有业务输入非法。

性能归因结合源码改动、同 source profile 事实和完整未开启 profile 计时。whole-capture 的 distinct cores、物理行数和声明 blocks 不足以推出“硬件没有充分利用”；数值 PASS 也不足以排除重叠写入的并发风险。

若 agent 因已知框架限制停住，区分工具链边界和 agent 失误。可以继续分析其他题目，不修改 PyPTO/runtime 来提高成功率。

## 校准阶段的发现

两条 8-step 短轨迹的第一个提案均在 3 个 step 内返回合法 JSON，通过源码 guard，并进入编译。都将 FP32/INDEX 除法修正为精确 reciprocal 常量，随后暴露 TensorType/TileType 写回错误。pfdb 组第二个提案改用 assemble，继续暴露 FP32 reduction 结果的 32-byte 对齐限制；file 组第二个提案改用显式 load/store 的 Tile 路径，随后因 row_sum 未提供所需 scratch 失败。这是错误推进和不同修复路径的证据，不能称为性能改善。

这里还发现一个实验输入限制：给模型的是四份 coding guide，不是可任意检索的全部编程资料。简明 operations 页只示例单参数 row_sum；更深层的[现有 reduction 参考](../pypto-language/15-reductions.md)已经明确区分 Tensor 与 Tile 的 scratch 契约。只读检查当前 PyPTO unified_ops/row_sum 的 overload 与实现也确认：Tensor 输入不能传 tmp，Tile 输入必须传满足 dtype/rank/extent 要求的 tmp。不是 runtime 缺少一个 profiling 分析。

v3/v4 不在中途改变这四份 guide；其 hash 在启动前冻结。若评估更完整的算子开发能力，应在后续独立实验中开放两组相同的编程资料检索，或补齐明确的共享参考。不能把本轮上下文裁剪导致的知识缺口直接列为 pfdb 的功能缺陷，也不应把整套 API 手册搬进数据库。

这个过程至少区分三层责任：

| 观察 | 当前归属 | 对 pfdb 的设计含义 |
| --- | --- | --- |
| 文件读取被拒绝 | 研究执行器权限配置 | 必须验证实际内容读取成功，配置看起来正确不够 |
| step 用尽后只返回总结 | OpenCode 客户端预算和提案协议 | 不能用这种格式失败证明数据库信息不足；需要给提案保留预算 |
| 能读到错误，也能消除错误，但出现下一层限制 | agent 的 DSL/硬件约束处理与编译反馈链 | 保留 source 对应的诊断与父子候选链；不能把缺少 PMU 当作编译修复失败原因 |
| 无 run 时仍查询空 run/baseline | agent 导航与工具选择成本 | 应衡量无效读取比例；单次未用不等于工具冗余 |
| 数值验证范围和 build provenance 不完整 | 外部验证/producer 的声明边界 | 数据库应显式保留缺失状态，不能补成“正确、安全、通用” |

数据库已经能交付对应 source 的错误正文，本轮没有证据要求新增另一个“诊断工具”。更有价值的下一步是，在可行闭环中比较诊断读取成本、合法候选产出和独立确认结果，再决定是否改工具导航。当前不删分析能力，也不根据编译失败增加猜测性的瓶颈标签。

## 账本与格式审计

结果账本由 summarize_native_operators.py 从实际提案记录、源码 hash、客户端事件和后端 trace 重建。未完成轨迹保持 incomplete；数值通过不自动成为安全候选或性能胜利；当前 source 的诊断读取与历史诊断读取区分计数。主结果要求总账、逐轨迹和逐提案一致。

该汇总脚本在 v3 开始后增加，属于事后分析，不参与模型输入或采集。另行记录一种协议失败：模型可能在合法 JSON 前加说明文字。主轨迹仍按冻结的严格解析规则拒绝；事后审计只检查是否能唯一抽取满足字段类型要求的 JSON，不执行、编译或加入这份提案，也不改写历史。含多个候选或 provider error 时不算可恢复格式。这个字段只说明被拒响应中存在唯一且类型完整的提案，拒绝也可能涉及读取权限等其他边界；它不能证明代码正确或更快。

最终选择须安全审查、已有额外 fixture 与独立 100/5 计时。v3 的 HC 确认结果在下文单列；v4 因存储错误中断；已经完成的确认在下文单列，未确认的搜索结果不能视作收益。

## 工件与复现

完整日志、私有数据库、生成候选和协议快照在忽略目录：

- build_output/pfdb_development/native-operator-v1/：bootstrap 与停止的配置错误批次。
- build_output/pfdb_development/native-read-permission-calibration/：真实读取校准。
- build_output/pfdb_development/native-operator-v2/：冻结协议、逐会话事件、原始采集和因预算上限停止的批次。
- build_output/pfdb_development/native-budget-calibration/：已完成的四次短轨迹校准。
- build_output/pfdb_development/native-operator-v3/：已结束但受到服务中断的对照、冻结协议和 HC 独立确认。
- build_output/pfdb_development/native-operator-v4/：服务恢复后启动、因 ENOSPC 中断的独立对照、冻结协议及独立确认。
- [可行性摘要](../../profile_db/research/pfdb_development/native_operator_feasibility.json)：不包含机器路径、模型 transcript 或隐藏推理的缩减账本。

入口是[外部执行器](../../profile_db/research/pfdb_development/run_native_operators.py)。它拒绝覆盖已有轨迹，不在失败后自动补跑。付费模型与设备测试不放进 CI。同目录 test_native_operators.py 和 test_native_interface.py 只验证确定性契约，不触碰设备。

## v3 终局：服务失败、可保留候选和独立确认

v3 的 16 条轨迹均结束，共观察到 64 个会话，但 46 个出现 provider error：3 个 provider.auth，43 个 provider.no-route。认证消息是 Reconnect OpenCode Console to continue；批次结束后模型目录不再包含 OpenCode Go。用户重新连接后，指定 DeepSeek 模型已在目录恢复。不能把这 46 个会话列为算子开发失败，也不能仅剔除它们后宣称剩余样本代表两组效果。

| 主轨迹记录 | native_file | pfdb_diagnostic |
| --- | ---: | ---: |
| 计划/已启动会话 | 32 | 32 |
| 被原冻结解析器接受的提案 | 7 | 9 |
| 进入外部编译/验证 | 7 | 9 |
| 搜索数值 PASS | 1 | 1 |

两个 PASS 都来自 HC 的第一次重复；第二次重复受服务中断影响。没有足够的有效重复估计入口效应。原始严格评分保留；事后能提取 JSON 的拒绝提案也不回填、编译或加入主轨迹。

静态检查保留了两个 HC 源码。native_file 候选按相邻两个 output head 融合，D_TILE=2048；pfdb_diagnostic 候选融合四个 output head，D_TILE=1024。它们都在 T=8 时声明 32 个 block，写入区间是 token、head、D 分块的互不重叠划分；组合系数的索引均为 4*k+h。初始候选声明 64 个 block。

随后独立确认四份源码：初始 Go、手写、两个新候选。均为 a2a3，由 npu-run 分配物理卡，逻辑设备0，重放同一份冻结 golden；每个源码三个独立进程，每进程100轮、warmup5，指标是完整未开启 profile 的 device effective latency median。

| 源码 | 三个进程 median，µs | 三个 median 的中位数，µs | 额外 seed17 fixture | 探索性 T=13 |
| --- | --- | ---: | --- | --- |
| 初始 Go | 22.1 / 22.2 / 21.6 | 22.1 | PASS | PASS |
| 手写 | 15.7 / 15.4 / 15.6 | 15.6 | PASS | FAIL |
| native_file 保留候选 | 18.75 / 18.7 / 19.0 | 18.75 | PASS | PASS |
| pfdb_diagnostic 保留候选 | 18.6 / 18.4 / 18.5 | 18.5 | PASS | PASS |

两个候选相对初始源码分别降低约15.2%、16.3%的延迟，但仍比手写慢约20.2%、18.6%。这是具体源码与工作负载的确认结果，不是诊断入口的因果效应。

手写的执行策略也揭示了比“更多核”“更少读取”更复杂的经验：它使用 T_TILE=4，将四个 token 放进一个 block，外层 token pipeline stage2，内层 HC reduction pipeline stage4；T=8 时只有8个 block。源码层面 x/residual 的显式读取量，初始与手写均约2.25MiB，两个新候选分别约1.125MiB和0.5625MiB，输出均0.5MiB。这些是源级读取量，不是实际 HBM 流量，未计缓存、spill 或编译器额外访问。手写读得更多却更快，表明不能用读取量或核心覆盖率独立排序候选；任务开销摊薄、流水与缓存需要另外验证，当前没有 PMU 或匹配消融足以分离它们。

T=13 也不能简单解释为手写错误：该手写路径按 T//4 分组，原配置对 token 数有整除约束；T=13 只覆盖12行。它与本轮两个 Go 候选的适用范围不同。T=8 是共同有效范围，因而上述计时仍可比较；不能把这份手写路径推广为任意动态 T 的通用基线。

## 已验证的信息遗漏：错误报告存在，但未进入交接

HC file 组第一个候选的第二路输出用了错误的组合系数：

- 正确行索引应为 0/1/2/3。
- 错误的第二路用了 1/2/3/3，影响 head1/head3。
- 原 stdout 只有4952字节，没有详细 mismatch；collection.json 有2662字节，记录60928/131072个 mismatch、首个位置4096及前20个 actual/expected。
- 默认诊断没有这些细节，原因不是16KiB截断，而是原 producer 未注册 collection.json。

对原错误表达式作 CPU 数学复算，得到各 head 的 mismatch 数 0/30455/0/30473，合计60928；前20个 actual 与已记录 NPU 值的最大差为0。它支持这是生成代码的索引错误，不需要解释成 runtime 精度问题。

只读工件审计保留原 manifest，新建一份只增加现有报告锚点的清单。已有 list_trials diagnostics 随即返回 mismatch 数和首个位置，错误 source SHA 仍拒绝正文。证明这里缺的是外部 producer 交接，不是数据库还缺一个诊断工具。file 组随后即使没有详细报告也自行修正了索引，因此尚不能说补报告在本例减少了修改次数。

本轮已落实最小修正：

1. 外部 producer/2 将非空错误报告注册为 source-bound diagnostic。
2. 对当前 golden validator 明确的 numerical mismatch，在 fixture 未变时按 failed 绑定，避免落入默认 not_run；其他异常不冒充数值失败。
3. 原生实验的规范诊断从全部注册诊断生成，使两个入口都获得同一份报告内容。
4. 每次模型调用前检查指定模型在目录中；遇到认证/路由错误后，停止各轨迹的后续请求。
5. 统一提案解析只接收一个字段和类型完整的 JSON 对象，可带说明文字；多候选、provider error、越界源码修改及实际未授权读取仍拒绝。

这些是外部执行器和交接契约修正，没有改 runtime、手写算子或数据库分析算法。对错误格式的窄识别规则及升级检查已记入[复用台账](pfdb-runtime-integration.md)。

## 服务恢复后的 v4

v4 另行冻结协议，从相同四题 bootstrap 重新开始，仍为两组、两次重复、每轨迹最多四个提案；不复用 v3 的 Go 提案。任务提交顺序先覆盖四种结构，第二次重复反转入口顺序。它包含上述交接与客户端修正，因而不能与 v3 混为一套不变处理的样本。

共享的四份编程指南仍不变，未给 agent 手写优化函数体；编译、NPU 和 profile 仍在数据库外。v4 后续因共享盘 ENOSPC 中断；已有54份完整提案记录和12条完成轨迹。8份保留源码的确认已完成，入口效果无法估计；详细终局见后文。

缩减结果见[完整 v3 账本](../../profile_db/research/pfdb_development/native_operator_summary.json)、[失败交接审计](../../profile_db/research/pfdb_development/native_failure_handoff_audit.json)与[索引复算](../../profile_db/research/pfdb_development/native_failure_indexing_proof.json)。它们不导出机器路径、完整 transcript 或隐藏推理正文。

## 主实验前的确定性验证

pfdb 与研究执行器的 CPU 测试共489项通过、11项跳过，保留一条既有 Pydantic 字段遮蔽 warning；新增交接、解析、服务停止和汇总测试包含在内。付费模型和设备实验独立执行，不混入这项测试通过数。

## 能力取舍的当前证据边界

本轮修正优先补齐事实流转，不增加分析入口。详细 numerical mismatch 已存在且已有 diagnostics view 能返回，因此新增一个“精度诊断工具”会重复现有能力；应修外部 producer 的注册和状态绑定。这个结论只针对这份明确遗漏，不能推导所有精度问题都已覆盖。

同样，RMS 的 Tensor/Tile 写回错误、reduction scratch 与物理对齐要求，主要需要准确的语言资料和编译反馈。缺失 PMU 不妨碍识别这些编译错误，也不能解释为 pfdb 需要实现一个代码生成器。HC 的正确融合、组合系数索引和不重叠划分则依赖 agent 将数学表达式转化为任务布局；数据库能够保存支持或反驳它的测量，不能凭低占用率保证这种变换正确或更快。

还要区分“查询未提供新事实”和“功能冗余”：单 task 的 task、why_long、critical_path 可能重复其 envelope，但同一接口在多 task 的 SWA/compressor 仍有不同作用。未采集 PMU、args 或 incore 时返回 unavailable 是缺失状态，不是这些分析永远没有价值。入口取舍必须结合问题类型、已有 artifact inventory、实际响应字节与候选结果，不按本轮调用频率删功能。

v4 的一次早期卷积提案还出现了可复查的导航浪费：agent 先查询 region=[0,25]，结果为空；实际 row 位于[97.32,118.18]，随后从 rows 取得了正确时间。overview 只有时长、没有时间区间端点，调用者容易把相对延迟误当成查询坐标。这一缺口已在 v4 中断并归档后的0.8.1修正；增加端点是否提高优化效果仍未证明。冻结实验输入没有改变。

另一个语义风险来自 HC file 组第4次提案：它从 PATH 的 zero stall 和 row envelope 推断标量读取、store drain 与内存争用，甚至用未测的向量运算耗时估计排除计算限制。这些是可测试假设，不是捕获已经证明的事实。runtime 的关键路径 analyze_rank 使用 frontier sweep：compute 是该任务 start/end 区间对路径的非重叠贡献，stall 是任务前的区间缺口；单 task 的 stall=0 不排除 task 内等待、不同 block 之间的空隙、pipe stall 或访存等待。pfdb 当前 why_long 会提供生命周期/pipe 活动的区别，但只读取 critical_path 的调用者不一定得到这项定义。本轮0.8.1随后在原入口补齐了关键度量定义，没有修改 runtime 数值或输出自动瓶颈判断。

已结束的 conv file 轨迹还有一个确定性的缺失状态遗漏：首次 why_sparse 只返回 ANALYSIS status=ok，没有任何 counter-window 结果。只读检查同一私有数据库的 scheduler snapshot，counters=[]、window=[97.32,118.18]。分析执行成功不等于存在可积分的计数器，空结果也不能解释为计数为0。这里应保留 runtime 的结果，并由 pfdb 明确返回所选窗口/engine 没有可用 counter track 的 unavailable 状态；无需修改 runtime 或新增分析算法。该修正已在 v4 中断归档后实施，冻结实验包保持原样。

## v4 已完成轨迹的独立确认：RMS 正例与精度消融

v4 pfdb 组第1条 RMS 轨迹最终得到可编译且数值通过的源码。修复链是 INDEX 除法→Tensor/Tile 写回→FP32 reduction tile 的32B对齐→循环累加器作用域。最终将8个真实 token 放进一个 block，D=4096 拆成4个1024列片段，两遍读取；先合并所有平方和，再计算归一化并写完所有列。第三次提案中循环 carry 不可见，第四次通过展开4段规避了该作用域问题。这是模型自己提交的源码，没有人工修复。

在额外 T=128、seed17 fixture 上通过；独立三个100/5进程 median 为15.2/14.6/15.3µs，中位数15.2µs。source/golden 匹配的手写归档控制为20.65/20.9/20.6µs，中位数20.65µs。二者都以8行分组，探索性 T=13 均不通过，不能视为任意动态 token 数的实现。这个正例发生在第一次取得有效 profile 之前，因此不能把收益归因于读取了 PMU、关键路径或片上指令分析。

手写路径使用128列 tile、两个各32段的 stage2 pipeline，并启用高精度 rsqrt；agent 使用1024列、两遍各4段的显式展开，默认 rsqrt 精度。它们在 decode T=8 下均只有1个 block。更高的并行核心数不能解释这个结果，片段大小、循环形式与精度设置都需要分别考虑。

另外发起1次独立 Go 提案，只允许给现有 pl.rsqrt 增加 high_precision=True；AST 检查要求其余整个源码结构完全相同。高精度版本仍通过额外 prefill fixture，确认 median 为15.6/15.55/15.4µs，中位数15.55µs，仍低于手写归档控制。因而不能将全部改善归为降低 rsqrt 精度。这个消融没有分离 tile 宽度与展开形式，也没有成对交错测量，不能把两份 agent 源码约0.35µs的差全部归因于精度开关。

该独立消融初次在模型调用前被过严的权限尾部断言停止，0次模型调用；修正研究执行器允许 wildcard deny 后跟 browser deny 后，换新目录执行。原失败及日志保留，不计为算子失败。缩减账本见[精度消融](../../profile_db/research/pfdb_development/native_rms_precision_probe.json)。

本节的手写计时是复用的既有独立归档，保持 source 与 golden hash 一致；没有重新采集控制以追逐有利数字。不同批次的设备/服务状态仍是限制。更快的是该题固定 EXAM 接口下的独立算子，不能直接替代手写版本在完整模型中的 TaskId/early-resolve 协同用途，也没有测量完整模型收益。主对照因存储错误未完成，不能由这个单题正例宣布诊断入口普遍有效。

## v4 终局、完整账本与存储中断

预注册上限64次提案，实际观察到58个会话，其中54份提案结果完整落盘、12/16条轨迹完成。余下4条在写文件时遇到 ENOSPC，4个已观察会话没有完整提案记录，另6个槽位未启动。没有认证/路由错误；不能把未记录或未启动的槽位算作算子失败。原始结果不补写，不重放丢失的提案，也不把不完整两组比较转成成功率结论。

| 已记录结果 | native_file | pfdb_diagnostic |
| --- | ---: | ---: |
| 完整提案记录 | 23 | 31 |
| 取得当前 source 的诊断 | 23 | 31 |
| JSON/源码 guard 接受并外部验证 | 18 | 31 |
| 数值 PASS | 5 | 14 |
| 提案协议拒绝 | 5 | 0 |

两个入口的原始诊断内容相同，但工具封装不同：原生 read 增加行号，pfdb 使用带身份和证据字段的事实行。处理也包含该差异。文件组的5次拒绝是提案字段不符合冻结协议，不能因此证明其代码无优化价值；也不能事后忽略额外字段，把这些代码补进主轨迹。样本少、重复不完整、各类错误分布不均，表格只记观察结果，不估计入口因果效果。

按题目分解54份完整记录：

| 题目 | 进入验证 | 数值 PASS | 明确数值 FAIL | 编译或进程失败 | 提案拒绝 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMS | 13 | 2 | 0 | 11 | 3 |
| HC | 13 | 9 | 0 | 4 | 1 |
| Conv | 10 | 8 | 0 | 2 | 1 |
| MTP | 13 | 0 | 5 | 8 | 0 |

“编译或进程失败”不假装有结构化阶段证据。MTP 的5份数值失败均为11482/131072点超差，8.7601%超过允许的5%。模型曾把它解释为近似 reciprocal/rsqrt 或正常量化噪声；最大误差小、分布接近，不足以证明这种归因。

58个会话的 aggregate usage metadata 均取得：input4,139,878、output174,842、reasoning count814,063；provider-native cost1.224228444。这是元数据，不代表订阅实际扣费或货币，且不能只统计完成轨迹而隐藏中断会话。

存储恢复只处理本研究自己的工件：32978个文件、3,989,572,647字节，在系统盘保留内容及mtime均经校验的副本；源码和数据库恢复在原物理目录，331个冻结 tensor 文件以链接使用归档副本。没有删除其他工作数据。部分写入的新测试编辑已恢复，原有测试保留。工作盘释放约2GiB，但仍接近满盘；临时归档必须保留，系统清理前需要迁回持久位置。本地 relocation/storage-layout 清单记录物理位置，公开账本不含机器绝对路径。

## 八份源码确认与当期手写控制

确认均为 a2a3、npu-run 分配物理卡、逻辑设备0；同题重放同一份冻结 golden；每份源码三个独立进程，每进程100轮、warmup5，以三个 device effective latency median 的中位数表示。8份源码均通过额外 fixture；RMS 是 T128/seed17，HC 是 T8/seed17，Conv 是已有 single decode fixture。三个手写控制另外各做一次100/5当期漂移检查，不能把其单进程与三进程统计混同。

| 题目 / 来源 | 三个进程 median，µs | 中位数，µs | 当期手写控制单进程 median，µs |
| --- | --- | ---: | ---: |
| RMS pfdb repeat0 | 15.2 / 14.6 / 15.3 | 15.2 | 20.6 |
| RMS pfdb repeat1 | 15.2 / 15.7 / 15.5 | 15.5 | 20.6 |
| HC file repeat0 | 18.7 / 18.75 / 18.75 | 18.75 | 15.9 |
| HC pfdb repeat0 | 18.8 / 18.8 / 18.65 | 18.8 | 15.9 |
| HC pfdb repeat1 | 17.8 / 17.7 / 17.95 | 17.8 | 15.9 |
| Conv file repeat0 | 14.15 / 13.9 / 14.0 | 14.0 | 14.4 |
| Conv pfdb repeat0 | 12.9 / 12.75 / 12.9 | 12.9 | 14.4 |
| Conv pfdb repeat1，轨迹中断前保留 | 13.3 / 13.55 / 13.7 | 13.55 | 14.4 |

最后一行只从中断前3份已验证提案选择，不作为完整四提案轨迹参与组间比较。所有 source/golden hash、选择前的源码审查及确认结果 hash 见[完整 v4 缩减账本](../../profile_db/research/pfdb_development/native_operator_v4_summary.json)。原始日志和编译产物留在忽略目录。

源级策略与限制：

- RMS 两份源码使用同一类8行/1024列/展开4段策略，源文件不同但不能算两种独立机制。decode 下均为1个 block，较快结果不能解释为更多核心。高精度开关消融后仍较快，支持继续研究更粗的内层工作单元；尚未分离 tile、展开和流水因素。
- HC 三份源码都在32个 block 左右融合或扩大粒度，仍慢于8个 block 的手写 token pipeline。读取更少、核心更多、单波执行都不足以独立排序。pfdb repeat0 的 sync_start 额外要求所有块同时有槽位；T13 的52个块被 runtime 的 REQUIRE_SYNC_START_INVALID 正确拒绝，这是适用范围限制，不是 runtime bug。另两份 HC 的 T13 通过。
- Conv 从1个串行块改为3或6个独立列分区，并保持每分区 token 顺序和全部 InOut 更新。最快源码将三行 state/weight 合并读取与归约，输出采用平坦视图；这些因素同时变化，不能只归功于某条 compiler hint 或并行度。6块可以快于手写15块，并不支持无限细分。

## MTP 的编译器缺陷与真正值得补的信息

MTP 的已生成 PTO 表明，源代码的 FP32→INT8 rint 被合法化为 FP32→FP16 RINT→INT8 RINT。独立8×32的最小 device probe 使用 +/-1.4999、+/-2.5001：直接 cast 的256点全部不符 torch.round，显式先 INT32 RNE 再经 FP16/INT8 的控制路径全部通过。PyPTO 的 LegalizeTileCast 选择该链并把 mode 传给每一跳；runtime 正确执行了生成指令，错误归属不能转给 runtime。

该缺陷已在本地 KNOWN_PYPTO_ISSUES.md 登记，可复用的最小复现为 profile_db/research/pfdb_development/probe_fp32_int8_rounding.py（与实际执行的本地probe的kernel AST一致），原日志与 PTO 位于 build_output/pfdb_development/fp32-i8-double-round/。没有修改 PyPTO/runtime，也没有把调查者的控制路径修补进 Go 的主实验候选。深层 precision guide 的直接 cast 建议已补上当前 a2a3 工具链限制；四份主实验 coding guide 保持冻结。

把隐藏 FP16 hop 在 CPU 上复算，会产生11750个超差点，而原 NPU 报告为11482；前10个报告点也不是完全相同。因此最小 probe 确认的是一个真实编译器语义缺陷，尚不能声称已经解释了 MTP 的全部误差。进一步定位需要比较真实量化中间值、scale 和源级数学；现有 args_dump 能力可承载相应已采集工件，但本轮没有这些捕获，pfdb 不应补造。

这比“再加一个利用率指标”更接近实际信息缺口：agent 已读到源级 cast 与最终 mismatch，却看不到该 source 对应的隐藏转换链。既有编译产物中存在相关信息，下一步应检验按 source/函数定位的少量 lowering 证据是否能改善修复，而不是默认把整份 IR 或 C++ 塞给 agent。

## 已落实到0.8.1的取舍与停止边界

| 有直接证据的问题 | 本轮最小修正 | 保持的边界 |
| --- | --- | --- |
| region 查询把时长当坐标 | overview 给出捕获 execution row 的 TIMELINE | 不归零重算 runtime 时间，不改变 region 选择语义 |
| 将零 path stall 误作无内部等待 | critical_path 先给生命周期/compute/stall定义 | 原 PATH 数值和官方算法不变 |
| 空 counter 只剩分析成功头 | why_sparse 明确 OVERHEAD_WINDOW unavailable | 真正存在的零值 series 仍是 measured zero |
| 未绑定报告默认 not_run | 未声明 validation 默认 unknown，显式 not_run 可用 | 旧 manifest 不回写，passed/failed四锚点不放宽 |
| 详细 numerical report 未交接 | 外部 producer 注册现有错误报告，producer/3使用新缺失状态 | 数据库不采集、不执行、不判断“怎么优化” |

完整 CPU 检查497项通过、11项跳过，保留1条既有 Pydantic warning；针对性84项通过，lint与快照同步。保护集293份源文件/原数据库与两个上游仓库均保持一致。没有新增顶层工具，数据库schema9、MCP schema7、只读33项/可写41项目录不变。复用与升级锚定见[维护台账](pfdb-runtime-integration.md)。

本轮没有足够依据继续删除分析接口，也没有证据支持新增自动瓶颈判断或自动优化建议。未调用、重复返回和缺失捕获应分别处理；单task的重复 envelope 并不证明多task分析冗余。生成代码定位、按阶段提供适用工具、更多编程资料检索是后续假设，需要另行固定问题与预算；不能以本轮中断样本宣布它们必然有效。

最小舍入复现的缩减证据见 [native_cast_lowering_probe.json](../../profile_db/research/pfdb_development/native_cast_lowering_probe.json)。另设的一对独立单步 Go 提案已完成，详细结果见下一节。它不属于已中断的原生对照，也没有接收手写实现、调查者的 bridge 代码或最小 probe 输出。

## 定向信息消融：有用的信息，已有入口也能承载

对同一份 MTP 失败源码、同一份11482点 mismatch 报告、同样四份 coding guide，分别发起一次工具全禁用的 Go 提案。只在其中一组额外提供原产物中3行未经解释的 PTO：FP16临时tile及两次RINT转换，保留source location。选择片段发生在调用前，是调查者针对该病例的定向选择；没有给模型 INT32 bridge 解法、手写函数或最小probe结果。

| 条件 | Agent 的修改 | 原冻结输入 | 新 T128/seed17 fixture |
| --- | --- | --- | --- |
| 只有源码与最终 mismatch | 对reciprocal/rsqrt做Newton refinement，保留直接INT8 cast | FAIL，11750/131072点超差 | 不执行 |
| 另加现有 lowering 片段 | 在INT8转换前先对FP32做INT32 RNE | PASS，原门槛不变 | PASS，原门槛不变 |

第二组由 Go 自己提出并写出源码，通过相同的 EXAM-body guard，没有人工修复。它与主实验的19次PASS分开记账，不追认进已中断轨迹。第一组的11750点结果恰与更准确FP32数学后的CPU双重舍入复算一致，说明“提高reciprocal精度”可以改变误差却依然保留关键错误。第二组的桥接仍可能含FP16，但发生在整数舍入之后，不能照搬模型“移除了所有FP16中间量”的说法。

这是一份定向机制证据，不是总体效应估计：每组只有一次提案，模型没有固定不可变revision，调用没有重复；调查者知道该看哪3行。不能据此宣布通用IR导出功能会提高agent成功率，也没有给这份新MTP源码做延迟验收或任意动态形状认证。[完整缩减账本](../../profile_db/research/pfdb_development/native_lowering_information_probe.json)保留source/protocol/result hash与两次调用usage。

随后用现有 pfdb API 做纯CPU交接验证：保留原manifest，新建派生清单，将现有 enorm_quant.pto 同时登记为 ir 元数据和选定 diagnostic。原 list_trials(view=diagnostics, source_sha256=...) 在16KiB有界尾部中返回了隐藏转换链，错误source SHA不返回正文，原清单hash不变。这说明本例首先缺的是**选定编译上下文的交接**，不是缺一个全新读取工具。证明见[现有入口交接账本](../../profile_db/research/pfdb_development/native_compiler_context_handoff.json)。

因此本轮没有新增IR工具、通用文件浏览器或独立的cast分析器；只在[候选证据说明](pfdb-candidate-evidence.md)补充复用当前diagnostic的方式。不能自动把所有IR都追加到默认反馈：本例有人定位函数，其他图的片段可能在中间、超过字节预算、或者来自不匹配构建。自动定位/选择、更多算子上的收益与上下文成本尚未得到验证；这是目前无法据现有证据直接决定的后续开发取舍。
