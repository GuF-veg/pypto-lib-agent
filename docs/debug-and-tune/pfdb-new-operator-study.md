# 从五道新题重新检验 pfdb 的信息价值


后续实施：诊断 view、执行计数和外部 manifest 交接已落实到 0.7，并完成六个实际 MCP 查询会话；见[原生接口验证](pfdb-native-interface-results.md)。这些是后续接口验证，未计入本文的 21 次主提案或性能比较。
日期：2026-10-22。本文承接[开发实验结果](pfdb-development-results.md)、[后续反例研究](pfdb-followup-results.md)和[候选证据设计](pfdb-candidate-evidence.md)。本轮只增加研究脚本、实验记录和文档，没有修改 pfdb 生产接口、PyPTO/runtime 或手写算子。

## 1. 这轮回答了什么

**本轮反复出现的困难，是把正确证据关联到当前候选、当前阶段和适用范围；单纯增加性能指标，没有保证模型写出更快、可接受的实现。** 本轮实际生成并在设备上验证了五道新题，不再局限于 compressor_ratio4 和 decode_sparse_attn_swa。

主要结果如下：

- lookup 的 agent 实现比手写控制更快，但保留方案只启动 8 个 block，手写版本启动 48 个。更多活跃核心不自动意味着更低延迟。
- HC post 的 pfdb 反馈方案比初稿快，仍慢于手写版本；手写版本只有 8 个 block，保留的 agent 版本有 32 个。
- convolution 初稿把计算集中在一个核心。基本反馈和 pfdb 反馈都能引导改进，但本轮保留的基本反馈方案更快。不能把优化成功全部归因于 pfdb。
- RMSNorm 和量化 MTP projection 在实现预算内没有产生可测的通过版本，涉及编译反馈、类型/形状理解以及工具链限制；MTP 的 M=8 epilogue 错误已用独立最小程序复现。此时再提供 PMU、关键路径或占用率没有直接帮助。
- 当前 pfdb 已经拥有不少被遗漏的信息：SPMD block 数和物理 rows 可由现有 `task`、`rows` 查询得到；候选身份、验证范围和状态演进声明可以放入现有 0.6 manifest。应先改进生产方接入和默认组织方式，再讨论增加数据库能力。

原始记录保留在忽略目录 `build_output/pfdb_development/new-operators-v1/`。可共享、无机器路径的[汇总数据](../../profile_db/research/pfdb_development/new_operator_summary.json)包含候选 SHA、三组计时、模型用量、实际反馈包规模和源文件保护检查。本文不把一个小规模探索试验当作模型或 pfdb 的总体成功率评测。

## 2. 实验设计与边界

### 2.1 题目与预算

从题库按行为差异选题，未依据手写版本的性能挑选：

| 新题 | 主要行为 | 搜索用例 | 额外现有用例 |
|---|---|---|---|
| [RMSNorm](../../benchmarks/basic_operator/deepseek_v4_flash_mtp/rmsnorm.py) | 归约、广播、BF16 输出 | T=8,D=4096 | prefill T=128 |
| [embedding lookup](../../benchmarks/basic_operator/deepseek_v4_flash_mtp/lookup_embedding.py) | 间接寻址、连续搬运 | T=8,D=4096,vocab=256 | prefill T=128 |
| [HC post](../../benchmarks/basic_operator/deepseek_v4_flash_mtp/hc_post.py) | FP32 残差混合、输入复用 | T=8,H=4,D=4096 | 原题只有 decode；更换 seed 为 17 |
| [KDA convolution decode](../../benchmarks/basic_operator/glm5_3_flash/kda_conv_decode.py) | 状态读写、逐通道卷积、SiLU | T=5，重排状态池行号 | decode-single T=3 |
| [MTP projection](../../benchmarks/basic_operator/deepseek_v4_flash_mtp/mtp_projection.py) | RMSNorm、INT8 量化、NZ 权重、GEMM | T=8,D=4096 | prefill T=128 |

每题最多 3 次实现提案。首个通过版本作为共同起点，分成两条分支，各最多 2 次修订：

1. **basic**：源码、相同语言指南、验证结果、编译错误和无插桩搜索计时。
2. **pfdb**：上述材料，加上同源码/搜索用例的 `overview`、`inventory`、`critical_path`、`perf_hints`、`memory`、`density`。后续通过版本重新收集已有种类的工件，再由 pfdb 离线读取。

失败消耗预算，不进行人工内核修复，不用其它题替换失败题。额外进行了 2 次编译诊断内容对照，共 **21 次主试验提案 + 2 次诊断提案**。实际调用固定为 OpenCode v2.1.2 的 `opencode-go/deepseek-v4.1-flash`，没有切换提供方。

### 2.2 这里的“agent”具体指什么

本轮是**外部编排的算子生成与修订试验**。OpenCode 收到完整题目、四篇公共编码指南和反馈，返回 EXAM 函数体；工具被禁止。外部 harness 检查修改范围、调用编译器、申请 NPU、验证数值并返还结果。每次提案是新会话，通过提示提供历史。

因此，它比静态问答更接近真实算子开发，但**没有测量自主搜索文档、发现 MCP 工具、选择查询或维护长会话的能力**。本轮也没有比较完整的 41 工具目录与简化目录。不能把模型没有看到的查询结果说成它忽视了证据，不能把所有失败归因于 pfdb。

主优化包使用 run 查询和显式 source SHA；没有把 0.6 候选 manifest 作为主试验的一项消融。诊断对照实际使用了 selected trial 接口，之后又进行了四组真实工件的 0.6 离线回放。两类证据的作用需要分开。

执行代码：[生成控制器](../../profile_db/research/pfdb_development/run_new_operators.py)、[外部验证器](../../profile_db/research/pfdb_development/collect_new_operator.py)、[确认控制器](../../profile_db/research/pfdb_development/confirm_new_operators.py)。原注册、逐次提示及响应、修订说明和与注册 SHA 完全匹配的初始脚本归档均保留在原始目录。

### 2.3 验证与计时口径

平台为 a2a3，单卡逻辑 device 0，由 `npu-run` 分配物理卡，没有固定同一物理卡。实际 capture 记录 20 AIC、40 AIV。搜索使用 20 轮、5 轮 warmup，仅用于选择候选。

每条分支在数值通过的候选和初稿之间，按搜索 median 选择保留源文件；相同源文件合并确认。最终对 12 个源文件——5 个手写控制、7 个 agent 实现——运行额外现有用例，并各做 **3 个独立进程、每进程 100 轮计时、5 轮 warmup**。确认复用冻结的搜索输入/golden 和同源码编译工件的私有副本，保留 source、build tree 和 golden tree 指纹。36 个确认进程全部通过。

下文数值为三次独立进程各自 `effective_us` 中位数的中位数，单位 µs；括号为这三个中位数的最小至最大值。它不是置信区间，也不是 profile makespan。所有 12 个保留源文件通过了各自的额外现有用例；这不代表所有中间提案均通过了这些用例。

状态型 convolution 有额外限制：正确性 dispatch 先验证初始状态，后续 warmup/计时继续更新同一个 `conv_state`，不在每轮恢复 pristine 状态。固定 mixed-QKV 和 5 次 warmup 下，3 行历史已经演进为重复当前 token。其计时代表这一重复 decode 条件，不代表每轮独立重放原始随机历史，也不表示逐轮输出都经过验证。依据是 `golden/runner.py::_benchmark_and_report` 和当前 runtime 的 benchmark 循环；没有修改它们。

## 3. 实测结果

| 算子 | 手写控制 | agent 初稿 | basic 保留方案 | pfdb 保留方案 |
|---|---:|---:|---:|---:|
| lookup | 18.90 (18.80–19.40) | 13.30 (13.25–13.50) | **11.20 (11.00–11.30)** | **同一个源文件，合并确认** |
| HC post | **15.80 (15.40–16.10)** | 23.70 (23.50–23.75) | 保留初稿 | 17.60 (17.50–17.90) |
| convolution decode | 14.60 (14.50–15.00) | 24.85 (24.70–24.90) | 14.40 (14.35–14.85) | 15.50 (15.40–15.50) |
| RMSNorm | 20.65 (20.60–20.90) | 3 次未编译通过 | 未进入修订 | 未进入修订 |
| MTP projection | 83.00 (82.70–83.10) | 3 次未编译通过 | 未进入修订 | 未进入修订 |

lookup 保留的 agent 源文件在本次 decode 工作负载下约为手写控制的 **1.69 倍速度**，延迟下降约 40.7%。HC 的 pfdb 方案比初稿下降约 25.7%，仍比手写控制慢约 11.4%。convolution basic 与手写控制的进程中位数范围重叠，1.4% 的中心值差异不足以宣称稳定超越；pfdb 保留方案也没有胜过 basic。

这里不报告“pfdb 平均提速百分比”：只有 3 个可配对优化任务，每题只有一个生成起点，且实现/反馈顺序固定、模型没有可复现的服务端 seed。数值表说明这些实现的结果，不能建立总体因果效应。

### 3.1 lookup：更粗的工作单元反而更快

[手写版本](../../models/deepseek_v4_flash_mtp/lookup_embedding.py)固定启动 48 个 SPMD block，用 512 列的小块分配 `T*(D/512)` 个 work item。T=8 时有 64 个 work item。agent 初稿改为 16 个 block、每块搬 2048 列；最终保留方案为 8 个 block、每块搬完整的 4096 列。

这些方案都只有 1 个逻辑 task。已采集的物理记录分别是 48、16、8 rows；对应无插桩延迟依次下降。源码变化同时减少了 block、标量 ID 读取和切片次数，并扩大连续搬运粒度。它们是与结果一致的机制解释，尚不能把收益精确拆成“调度节省多少、搬运节省多少”。需要固定 block 数、只改变每块内切片方式的后续对照。

这个正例支持的信息不是“发现低占用后增加 block”，而是**工作量、block 粒度、重复索引/切片和固定调度成本之间的关系**。basic 和 pfdb 两条分支最后保留了完全相同的源码，不能把该实现差异归功于某一反馈分支。

### 3.2 HC：复用有帮助，追满核心数仍会走错

agent 初稿把 token、out_h、D 分片全部展开，产生 64 个物理 rows。pfdb 的第一轮修订把同一 D 片段的 4 个输出放到同一个 block，复用 x 和四个 residual 片段，减少为 32 个 rows。这个方案确有更低的确认延迟。

但[手写控制](../../models/deepseek_v4_flash_mtp/hc_post.py)只产生 8 个 rows：每块处理 4 个 token，用完整 D 行和软件流水处理 token、输入 HC。它仍然更快。现有结果说明复用与任务粒度值得考虑，不能证明瓶颈已经是 HBM，也不能从忙核数直接推算 15–18 µs 的目标延迟；本轮没有相应 PMU 或固定成本模型。

第二轮 pfdb 提案为了从 32 个 block 增到 40 个，选择宽度 832，并把尾块起点钳制为 3264。相邻块对 `[3264,3328)` 的 64 个 FP32 列有重叠普通写入。默认数值测试通过，搜索 median 为 20.2 µs，比第一轮的 18.55 µs 更慢。

此提案记录为**并发写入安全性未证实，不作为可接受优化**。这不是已复现的 race 或 runtime bug；文档也并非一概禁止相同值的重复写入——[tensor system ops](../pypto-language/20-tensor-system-ops.md)在特定实测场景讨论过整行相同值写入。这里缺少的是对该部分重叠普通 TSTORE 的适用保证。改成不重叠尾部写入或取得明确保证后才适合重新验收。

安全审查没有为 pfdb 分支“换来”更好的计时：数值搜索原本已经保留第一轮方案，补上明确排除记录后重新执行选择，12 个确认源文件完全不变。排除清单最初一次写入因研究脚本路径处理失败未生效，此执行偏差及不变性复核保存在 `review-selection-audit.json`，原确认计划没有改写。

### 3.3 convolution：信息用对了，也不保证最终选择更好

初稿把 token 和三个 Q/K/V 分区串行放在一个 InCore region，实际 profile 只有 1 个 AIV row。basic 修订最终按状态池行和 512 列分区分配 block，池大小 6、每行 3 个分区，共 18 个 block；每个 block 顺序扫描 token，保留同状态行的更新顺序。

pfdb 第一轮修订使用 6 个 256 列 block，在每个列分区内保持 token 顺序，并把输出改为平坦视图，以回应实际出现的 PH001 短行 store 提示。这同时改变了并行度和 store 形状，不能孤立声称收益由某条 hint 导致。第二轮又尝试 token 并行和更细列块；搜索比第一轮慢，因此最终保留第一轮。

确认结果为 basic 14.4 µs、pfdb 15.5 µs，[手写控制](../../models/glm5_3_flash/kda_conv.py)14.6 µs。手写控制有 15 个 block。**1→6 个 block 的确改进了初稿，但 6、15、18 个 block 之间的排序不能靠占用率或单条 hint 决定。**

原题写明 one token per row；现有 fixture 的 `state_rows` 唯一且重排。利用这一合同进行 token 并行可以在该域内讨论。不能把未知重复行域默认纳入合法域，也不能反过来以假设的重复行把题目明确允许的优化一概判错。pfdb 应保存合同及证据范围，不替生产方或 agent 创造合同。

### 3.4 两道失败题：性能数据库无法替代语言和形状理解

RMSNorm 的局部整数 `d_tile` 在 DSL 中进入 index 类，`1.0 / d_tile` 触发 fp32/index 类型错误。两次早期反馈因外部 launcher 的 stderr 重定向而丢失错误正文，第三次仍失败；这些早期尝试必须标为反馈受损，不能解释为模型看到了同样错误却不修。

MTP 第一轮复用同一变量名存放不同形状/类型，修订后又在 matmul 输出的 16 行物理 tile 与 8 行 scale 之间产生 `row_expand_mul` 形状错误。第二、三轮均遇到后者，完整错误已经提供。

最终归因审查另外编写了[独立最小重现](../../profile_db/research/pfdb_development/probe_new_matmul.py)：逻辑 `[8,64] @ [32,64].T` 的 INT32 结果转为 FP32，乘形状匹配的 `[8,1]` scale。它只使用文档中的 rank-2 matmul、cast、row-expand 组合，却在当前 PyPTO lowering 后得到同一 `tile rows=16 and row_vec rows=8` 错误。该检查为 CPU compile-only，没有执行设备或修改上游。

因此，**MTP 的后两次失败不能仅归咎于 agent 的数学或形状推理**：这里至少有一个已复现的编译器组合限制，而模型也没有选用当前工具链可用的显式物理形状/有效形状模式。需要的是版本对应的诊断及约束知识，不是更多 PMU。该独立归因程序没有替 agent 修复候选，原失败次数和未获得性能数据的事实均保持不变。

两类失败都发生在合法新 profile 出现之前。旧版本或手写控制的成功 profile 不能冒充新候选的执行证据；编译失败也不应被表达为“执行耗时为零”。

## 4. 诊断内容对照：缺口存在，但效果只走到下一层错误

在观察到 RMS 失败之后，单独注册两个新会话，各 1 次提案，输入同一份失败源码、相同指南和同一个 selected trial 的 pfdb facts：

| 反馈 | 实际行为 | 结果 |
|---|---|---|
| 仅当前 pfdb 的诊断 path/hash/available | 猜测 weight staging 或广播问题，仍保留 `1.0 / d_tile` | 重复原类型错误 |
| 另加已注册、hash 一致的诊断正文 | 正确定位除法类别错误，改成浮点常量表达式 | 越过该错误，随后在 TensorType/TileType 的 `store` 上失败 |

这证明了一个具体的**内容检索接口缝隙**：当前候选接口可锚定诊断文件，但不返回其正文。它不证明 agent 最终能编译通过；两次修复都没有通过，也没有性能结果。该对照为事后选择的单反例，不是总体效果估计。

同样不能说现有系统“无法提供诊断”：文件读取工具可以打开已注册路径，调用方也可用现有 trial `evidence_refs` 存放摘要。本轮禁止了工具，所以测到的是仅靠 pfdb facts 的限制。后续应比较**原生 agent 自行读取文件**与**pfdb 直接返回受限内容**，再决定数据库侧是否值得补充。

实现：[诊断对照](../../profile_db/research/pfdb_development/probe_new_diagnostics.py)。没有修改 launcher 或 runtime；只在外部验证子进程中把 stderr 保留到已捕获的输出，原始失败仍在。

## 5. 再检验 0.6：哪些已经能表达，哪些还缺

使用[离线回放脚本](../../profile_db/research/pfdb_development/replay_new_candidates.py)，把 lookup、HC、convolution 的三个保留方案和一个真实 RMS 编译失败导入新数据库；只读已有工件，没有新模型或设备采集。

三个成功案例均导入 source/build、合同文本、验证报告/参考/比较器/fixture、三组 bench，以及对应 profile 的 records/deps/name_map，并成功绑定 run。失败案例没有假造 build 或 run。四例均通过 source/scope guard 读取，错误 source guard 均被拒绝。

结论是：

- **身份、范围、缺失状态的底座可用。** `workload` 已可声明 convolution 的 `benchmark_state_policy`；验证范围已可说明只验证初始 dispatch 和列出的 fixture。无需先增加新表或新工具。
- **验证信息仍是生产方声明。** 有 authority 文件并不等于 pfdb 认证了其解释，PASS 不认证所有 timed output，更不认证并发写入安全性。
- **selected trial 的内容组织还有改进空间。** 三个成功案例完整 facts 为约 5.4–5.7 KB；当前预算恢复机制能取完整。这里测到了响应规模，没有测到这些字段影响 native agent 表现，不能据此宣布字段冗余。
- **compile-only 诊断仍是锚点而非正文。** 这与上一节的实测一致。
- **run compare 与候选范围仍要正确衔接。** 现有 compare 明确报告 `compatibility_scope=capture_configuration_and_profile_analysis`、`workload_validation_checked=false`，没有假装认证工作负载。若未来希望自动阻止错误范围比较，应复用已存在的 source/scope 绑定，而不是另建一套候选身份。

当前实现来源：[候选模型及事实](../../profile_db/src/profile_db/lifecycle/candidate.py)、[compare 的范围声明](../../profile_db/src/profile_db/lifecycle/compare.py)、[候选证据文档](pfdb-candidate-evidence.md)。本轮没有对这部分生产代码作修改。

## 6. 真正需要的信息，以及不能误判为缺陷的情况

### 6.1 每条有效反馈应能连起这条链

`当前源码/构建 → 工作负载与合同 → 验证范围 → 无插桩计时 → task/block/物理 rows → 当前问题需要的工件片段`

重点是关系，而不是把每张表都发给 agent：

| 开发问题 | 最小有用信息 | 本轮证据及归属 |
|---|---|---|
| 当前代码能否编译？ | 同源码的失败阶段、错误正文及来源；未执行状态 | RMS/MTP；内容检索是候选接口缝隙，编译知识属于 agent/指南 |
| 改动是否合法？ | 输入域、别名/状态/写入所有权假设、验证覆盖 | HC 部分重叠写入、conv 状态行；不是性能计数器能认证的事实 |
| 是否真正变快？ | 同工作负载、同 metric 合同、独立计时 strata、当前候选绑定 | 36 次确认；现有 bench/compare 能力应复用 |
| 1 个 task 是否意味着只有 1 核？ | block_num、物理 row 数、不同核心数、每核心 row 分布 | lookup/HC 逻辑 task 都为 1，但物理 rows 差异很大；已有查询覆盖 |
| 是否应该细分任务？ | 每块有效工作、重复读取、连续搬运粒度及调度成本线索 | lookup/HC/conv 排序不单调；块数不能充当优化答案 |
| 是否需要更多 profiling？ | 当前缺失的是未请求、未提供、解析失败，还是根本尚未执行 | lookup/HC 无 memory/hint 文件，不能说 pfdb 丢了证据 |

### 6.2 本轮没有证明哪些能力冗余

lookup/HC 的初始 capture 中没有 memory 或 perf_hint 文件，pfdb 的 unavailable 与工件一致。convolution 确有三条 PH001 短行 store 提示，模型实际使用了它们。PMU 未请求，不能据此删除 PMU；依赖图、scheduler、early-dispatch 在这些单 task 题中没有充分机会，也不能据此宣布冗余。

按逐次提示中的实际 JSON 计，初始六查询包约为 3.8、4.4、6.3 KB，分别约占完整提示字节的 6.8%、7.6%、9.1%。lookup/HC 中 inventory 占其中较大比例；完整提示还包含编码指南和题目，因此不能把整体上下文成本都归因于 pfdb。更合理的假设是**默认先呈现可行动的可用性摘要，再按需展开完整 inventory**，而不是删除库存信息。

`task`、`rows` 已能返回 SPMD 信息，初始六查询包却没有包含它们。这是反馈选择遗漏，不是数据库缺表。某些模型仍从源码推导出 block 数；后续应检验精简的 task/block 摘要是否比完整 rows 更有效。

另外，`density.busy_cores` 是在存储时间 band 内有任意重叠的不同核心数，聚合 bucket 又取这些计数的最大值，不是整个 band 连续忙，也不是同时峰值利用率。应保留这一解释，不能把 HC 的“40 busy cores”转换成“该时段全程 100% 利用率”。来源：[时间 band 构建](../../profile_db/src/profile_db/derived/time_band.py)、[density 查询](../../profile_db/src/profile_db/query/handlers_z1.py)。

### 6.3 “提供过”与“被正确使用”必须分开

两条 basic 响应声称某个源码 SHA 对应的实现未提供，但其完整函数体就在当前提示中；问题是源码、历史计时和当前状态的关联失败。部分 HC 响应还对 unroll 的并行发射或具体目标延迟作了缺少证据的推断。

这不直接证明 pfdb 0.6 身份接口无效，因为主试验并未把它作为消融。更直接的修复落点是外部生产方和消费提示：把 current source SHA、源文件、对应 trial/run、该候选 bench 放在一起，明确区分当前与历史证据。历史试验的后续 pfdb 包虽然已更新为新候选，提示标题仍写着 Initial candidate；应以保存的逐次 prompt 和 SHA 判断模型实际看到的材料，不能使用后来更新过的同名 packet 文件倒推。研究控制器现已修正后续提示标签，并把 current source SHA 紧贴源码；这个事后修正未参与上述测量。

## 7. 额外 T=13 挑战说明了范围管理的重要性

在确认前注册了一个额外挑战：对动态签名的非 conv 题，用现有 torch 参考构造 T=13、seed17。它不是原 CLI 现有 case，也没有用于延迟排名。

8 个保留源文件接受该挑战，5 个通过；失败的 3 个均为手写控制：RMSNorm、HC、MTP projection。它们在声明部署的 decode/prefill 用例仍通过，不能据此宣布该部署域发生回归。RMS/HC 源码中的整除分块及尾部缺失是可见的范围线索，但动态类型标注本身不能证明任意 T 都属于业务支持域。

因此：手写实现也是一个有范围的控制，不是任意输入的绝对 oracle；数值 oracle 仍是题目的 torch 参考。下一轮应先由维护者或已有规范确认支持域，再把整除条件、尾部约束和测试覆盖写进 producer manifest。这里不修改手写算子来扩大范围，也不把新挑战用于补齐或隐藏主试验的失败。

## 8. 后续开发优先级

### P0：先把已有能力接成可靠闭环

1. **生产方先接入 0.6。** 自动产出并导入真实 candidate manifest；把验证、benchmark 状态演进和 source/build 绑定随候选更新。pfdb 只接收、验证和查询工件，不启动采集。
2. **试验受限诊断内容读取。** 优先在现有 selected trial 入口返回 source-guarded、类型受限、预算受限的既有诊断片段；复用现有 budget/error/provenance。工具链限制附对应版本及已有规范/重现来源，区别于纯源码类型错误。先与 native 文件读取对照，再决定是否进入生产，避免每种日志新增一个工具。
3. **试验简洁的执行摘要。** 复用 `task`/`rows`，提供 task 数、block/row 数、不同核心数及分布，明确这些不是瓶颈判断。原始 rows 仍按需展开；不再新增另一套“利用率分析算法”。
4. **把结果资格与计时分开。** 数值 PASS、合同假设、未证实并发写入、额外范围测试分别记录。复用现有 validation/contract/verdict，不让 pfdb 自动认证正确性或宣判 win。

### P1：用原生工具使用验证，而不是继续只加反馈文本

建议下一轮保留 4 类题——编译失败的 RMS、形状复杂的 MTP、状态型 conv、复用/粒度冲突的 HC——各做 2 次独立起点，比较两个原生 OpenCode 条件，共 16 个会话，每会话最多 4 次候选提交，共最多 64 次提交。原生工具循环另设模型请求总硬上限 256 次，并记录 token 和时间；不能把候选提交数当作模型调用数。题目数和重复数分别报告。

第一组先固定可取得的事实，只改变“完整工具目录”与“围绕当前候选的精简工作集”，测量 agent 能否找到正确证据、请求/返回字节、编译修复、范围误用和最终有效延迟。模型可读白名单文档/候选/工件，连接只读 pfdb；外部 harness 继续管理验证和设备，runtime 仍只读。

第二组再单独比较“已有锚点 + native 文件读取”与“pfdb 受限内容”，避免把接口组织与新增事实混成一个处理。采用相同候选、固定提交和时间预算，保留失败、停止、工具错误、未通过的优化及当前源码错配。

默认信息的删减应在正向检验之后进行：证明某类信息有用，再移除它看是否损伤决策；PMU、memory、scheduler 等必须保留能体现价值的反例。现有五题不足以支持删除这些能力。

## 9. 可复查材料与验证

- [机器可读汇总](../../profile_db/research/pfdb_development/new_operator_summary.json)：21 次主提案、14 次搜索数值 PASS、12 个确认源文件、36 组确认数据、实际提示包规模、23 会话用量及身份回放。
- [汇总脚本](../../profile_db/research/pfdb_development/summarize_new_operators.py)：从逐次结果和不可变提示提取统计，并复查原文件/数据库与上游指纹。
- [研究边界测试](../../profile_db/research/pfdb_development/test_new_operators.py)：保护 EXAM 外代码、拒绝 provider/tool 异常、排除失败/不合格候选、选择与去重。研究 CPU 测试共 29 passed。
- 原目录中的 `task-audit.md`、`proposal-audit.md` 为独立只读源码/公开响应审查；没有导出隐藏推理正文，也没有让审查者替模型写内核。
- 293 个被保护的源文件/原数据库指纹一致；PyPTO `ee49fceaa3290cf90257a22eab867c04245b499d`、runtime `6e383fc57c12a8bd4dce045289b8b4f5d55a15f1` 的提交与 tracked diff 均未变。
- 23 个 OpenCode 会话的元数据均取得：input 153,585、output 37,251、reasoning 数量 448,923；API 原生 cost 合计约 0.3624。未取得其货币/订阅结算口径，不能把它解释为用户实际账单。

本文主试验尚未验证原生 MCP；后续只读工具发现和证据检索见[原生接口验证](pfdb-native-interface-results.md)。仍未验证：完整自主开发、服务端模型重复性、大样本因果收益、PMU/内存瓶颈归因，以及未声明的业务输入域。上述限制是下一轮实验的边界，不应通过给 pfdb 增加更多无来源的推断来填补。
