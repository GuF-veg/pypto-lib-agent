# pfdb 后续研究：输入前提、候选身份与工作关系

日期：2026-10-22。状态：本轮登记的实验与独立复核已完成，未改变 pfdb 默认接口。

本轮承接[上一轮实施记录](pfdb-development-results.md)末尾的三个优先项。
研究对象仍是 compressor ratio4 和 decode sparse attention SWA；不修改算子、
PyPTO/runtime，不修改原始数据库、capture 或旧实验的冻结输入。数据准备、
模型调用和设备验证都在数据库之外执行，pfdb 继续只分析已有工件。

## 要回答的问题

上一轮已经证明“字段答对”与“写出更快的合法算子”之间存在距离。新候选的
编译产物没有反馈给 agent；它可能沿用旧 hint，也可能将逻辑 unit、提交
block、物理 row、core 混为一谈。还有一些成功来自输入模式特化，而输入
初始化器生成的模式尚不足以授权一般化采用。

本轮分别考察三个边界：

1. **语义权限**：当前输入碰巧连续、已验证连续，与允许依赖连续性是三个
   不同事实。数据库能保存前两者；第三者需要来自输入域的权威声明。
2. **证据身份**：源代码改动后，哪些旧证据仍适用？编译、验证、计时与
   profiling 是否真正指向同一候选和 workload？
3. **实体关系**：已有原始值能否通过少量可追溯的关联降低误读，而无需
   新增一套 profiler、诊断引擎或大量面向 agent 的工具？

## 实验登记与解释规则

[冻结协议](../../profile_db/research/pfdb_development/followup_registration.json)
在本轮模型与设备调用之前写入。静态实验为六个问题、三个条件、每条件
三个重复，共 54 次调用；固定模型为 `opencode-go/deepseek-v4.1-flash`。
每次新会话、禁止工具及文件访问、并行度 2、超时 180 秒、不重试。服务方
未提供不可变模型 revision 或可复现 seed，重复编号不代表独立算子样本。

| 条件 | 提供的内容 | 可以比较什么 |
| --- | --- | --- |
| D0 | 起点/历史证据，明确缺失当前候选证据 | 能否正确拒绝沿用；unknown 是允许答案 |
| D1 | D0 加已有候选源码、编译片段、日志和身份锚点 | 新原始信息的恢复价值 |
| R | D1 原始内容完整保留，再加入源位置可核对的派生关系 | 显式关系是否进一步帮助解释 |

R 提供计算结果与关联，**不是纯格式对照**。契约问题包含原始证据相同的
负对照；不伪造 owner policy 或“普遍合法输入”来制造分差。固定字段与
引用检查、自由解释的证据支持、下一实验是否有效、缺失与成本分别记账。
静态实验不能证明设备加速或总体优化能力提升。

## 从主源码查明的事实

### 连续位置是证明前提，尚不是已查明的完整输入契约

canonical compressor 的 fixture 生成 `position_ids=arange(start,start+T)`。
扫描候选用 `position-position[0]` 反解物理 token 下标；这在 request 内
`p[j]=p[0]+j` 时成立。mapping 的物理页不连续本身不破坏这个等式。
只有“有序”不能保证等式成立：缺失一个 position 就可能漏掉窗口中的 token。

但是，当前已查到的文档和 TensorSpec 没有明确裁定 request 内跳过位置的
业务合法性。不能将“没写禁止”当作授权，也不能将 initializer 的 arange
当作一般输入域限制。已有连续 fixture 上的正确性与性能结果仍按原范围
保留；它们不自动升级成通用采用依据。

另一个限制来自 oracle：standalone golden 固定使用 state table 第 0 行、
按全局物理 tile 处理，没有读取 `query_start_loc`。本轮诊断因此只用一个
request，不用它裁决多 request。详见含源位置、集合推导和边界检查的
[输入契约审计](../../profile_db/research/pfdb_development/contract_followup_audit.md)。

### SWA 两版已有两个 Right 分配，变化在分配与迭代的对应关系

baseline K256 与候选 K128 的 `proj_a_mm` 都分配两个 32 KiB Right buffer。
baseline 的相邻外层 256-K chunk 各自拆成两个 128-K L0 slice，复用相同
Right 对；K128 的两个相邻外层 chunk 分别使用不同 Right 分配。因此单看
“两个 buffer”会丢失决定性的关联，不能写成从一个物理分配变成两个。

对应 PH-MR-001 从四条变为零是实际日志差异；生成物关系仍不能证明设备
流水已经重叠，更不能代替独立的端到端性能确认。历史 K128 搜索优势在
独立确认中没有保留，这也说明清除 hint 不是目标函数。

### 相同 compiled metadata 不能唯一标识 build

上述两版 `compiled_meta.json` 字节完全相同，SHA-256 均为
`a818d7044dfea0a1e694286ef88aa534e389d2b97e483fb339b56228a2534ac0`，
但 source、hint 和 MemoryReuse 文件不同。该文件说明相同签名与平台，
其 hash 不能单独证明源码/编译结果相同。现有外部确认器还核对 source 和
实际编译文件，不能将它简化为只比较 metadata。

### 逻辑工作、launch 与观测到的物理执行需要分开

C4 M112/N256/20 lanes 在第一个 512-row chunk 有
`ceil(512/112) × (1024/256) × 2 = 40` 个逻辑 unit，20 个 lane 各执行两次。
生成 orchestration 对 projection 是一次 AIC submit、block count 20。
首块 clamp 导致每个投影 N tile 重复覆盖 48 个 M 行位置。

原 baseline 有 32 条 projection 物理 row、分布于 20 个 AIC core。
这些不是候选的观测值：该候选没有 level-4/PMU，不能为它填入实际 row、
core 分配或活动率。Right/L0B 与 Vec/UB 也需保持独立；删掉向量临时量
不能直接证明 Right 容量被释放。

上述三个实体问题的文件 hash、原始片段位置、现有 pfdb 能力与 attach/trial
绑定边界见[关联审计](../../profile_db/research/pfdb_development/relations_followup_audit.md)。

## 四次设备诊断：确认了连续性前提的作用

从既有 confirmation registry 选取 K256 non-scan parent（source SHA 前缀
`c7ea6e9240a405e8`）与 K256 scan（`6fded29273bcc107`）。源码 diff 只有
扫描边界：全 tile 扫描改成根据 position 反解的局部区间。每次验证使用独立
复制的编译目录；不是在原 build 上 replay。原 affine512/start0 golden
完整复制；另一份输入保持相同单 request、shape、dtype 与权重，只令
`position_ids[3:]+=1`，联动重建 state/cache mapping 和 RoPE，再用未改动的
canonical CPU golden 生成一次新 expected 输出。

| 实现 | affine512 | 有序缺一个 position 的 512-row 诊断 |
| --- | --- | --- |
| non-scan K256 | PASS | PASS |
| scan K256 | PASS | **FAIL：仅 cmp_kv** |

scan-gap 的 `cmp_kv` 有 **63,501 / 134,217,728** 个元素超出原容差，
最大绝对差为 **2.07422**，坐标 `(1,14,0,218)`；首个报告的 flattened
坐标为 1024，actual `-0.984375`、expected `-0.9296875`。
虽然全 cache 的 mismatch 比例只有约 0.0473%，原 comparator 的
`max_error_ratio=0`，因此必须失败，没有放宽容差。

四次调用没有超时、没有重试，benchmark 关闭；外部子进程累计墙钟
74.790 秒是验证执行成本，**不是算子 latency**。这里的 exit 1 是保留的
数值校验失败，不是 pfdb 或 runtime 版本兼容错误。结果支持“此次变换
需要连续位置前提”，仍不能单凭本实验裁定有缺口位置属于支持域。
同时，non-scan 在该诊断通过也不是任意输入域的正确性证明。

执行器与边界检查见
[validate_contract.py](../../profile_db/research/pfdb_development/validate_contract.py)，
原始协议、四份独立结果与日志保存在忽略目录
`build_output/pfdb_development/followup-v1/contract/`；原始 snapshot 不变。

## 54 次静态调用：新证据有价值，关系事实的收益有边界

54 次尝试全部完成，无 provider error、超时、缺失响应或工具调用；54 份
session usage 均可读取。问题及答案索引冻结后没有按结果改题、改分或补跑。

首次启动在实际调用模型之前被预检拒绝。原因是旧执行器使用
`debug agents` 查询默认上下文，模型调用却指定隔离 workspace。
修正为只读 `agent.list` 显式传入 `location[directory]`、核对返回目录与
deny-all 权限后，在全新输出目录执行。六项预检 regression 通过；改动
没有触及 prompt、packet、scorer 或模型调用方式。旧失败目录和输入 freeze
保留，另存调用前 amendment，记录旧/新 runner hash。这是执行器预检修正，
不是模型失败重试，也没有改写旧研究的 runner 快照。

下面的通过要求是全部固定字段、引用 ID 和最低引用覆盖同时满足。
本轮引用检查全部通过，未通过均来自表中具体字段。

| 问题 | D0 / 3 | D1 / 3 | R / 3 | 直接观察 |
| --- | --- | --- | --- | --- |
| affine position 与物理 paging | 1 | 1 | 0 | 多次将物理 paging 重排误作 position 非 affine；R 未消除误读 |
| 一般输入域的 authority | 3 | 3 | 3 | 原始源码与范围提示已足以保留 unresolved；附加关系未增加字段收益 |
| SWA 当前候选 hint/分配使用关系 | 3 | 3 | 3 | D0 正确 unknown；D1 恢复 0 条 warning、两个分配、相邻 chunk 分别使用它们 |
| 相同 metadata 与不同 source/IR | 3 | 3 | 3 | D0 的文件指纹已经足够；再给源码片段/关系没有新增本题收益 |
| C4 unit/block/执行 row | 3 | 0 | 3 | D0 正确 unknown；D1 三次将输入 512 tokens 写成 observed execution rows；R 三次保持 unavailable |
| SWA Right/Vec 对应的硬件池 | 3 | 3 | 3 | D0 已知 Right/L0B、生成位置未知；D1 恢复 Vec/UB，R 没有新增本题收益 |
| **合计 / 18** | **16** | **13** | **15** | 条件的可回答内容不同，不能直接用总分判定是否应减少证据 |

特别是 C4：D1 的三次回答其实都算对了 **40 units、20 launch blocks、
2 steps/lane、48 个重复行位置、1 次 projection submit**。
出错点不是缺少除法公式，而是把 validation manifest 中的输入 tokens
当成 profiling 的物理执行 row。R 同时增加工作关系和明确的采集/可用性
边界，不能把改善单独归因于某一公式或图结构。

这支持一个比“多给生成物”更具体的候选设计：已有值仍保留，但分别命名
为输入大小、静态 launch 与实际观测；没有 trace 就明确说该观测字段
unavailable。只删除输入 token 数会丢失 validation 范围，不能解决关联问题。

affine/paging 问题共有七次失败。题目同时保留 affine 前提和原 mapping
guards，物理页的一致重排不会改变扫描候选集合。错误回答却引入
“non-affine consistent paging”等表述，混淆了两种独立变化。R 已给条件化
集合关系仍三次答错，因此不能把所有原始材料再补一条解释当作通用解法。
题目的英文措辞和否定式字段本身也可能增加误读；本轮不事后重写评分，
将这一限制保留到后续更自然的优化任务验证。

### 信息恢复、成本与到顶限制

SWA 当前 buffer、C4 静态工作、生成 Vec 位置共十个新增具体字段，每个
条件三次，共 30 个位置。D0 对这些位置正确报告 unavailable；D1 与 R 均
恢复全部 30 个具体值。D0 的 unknown 不是失败，也不是与 D1 相同难度的
答案。C4 随新材料出现的虚构观测应另列，不能被具体信息恢复掩盖。

| 条件 | 可见 prompt bytes（合计） | 尝试墙钟秒（合计） | provider native cost |
| --- | --- | --- | --- |
| D0 | 133,617 | 258.869 | 0.024716781 |
| D1 | 192,498 | 384.455 | 0.041653583 |
| R | 202,470 | 249.592 | 0.026038989 |

DeepSeek 实验 session cost 合计 **0.092409353**，未假定币种，未计研究与审查 agent 成本；墙钟合计
**892.916 秒**包含客户端/服务等待，并行度为 2，不能当作实际串行总工期。
成本仅读 session aggregate，不再加 message totals。原始 input/output、
reasoning token 计数与 cache read/write 分列保留，不读取或导出隐藏推理，
不自行拼成另一种 total-token 口径。

R 比 D1 的可见 prompt 多 9,972 bytes，约 5.18%；观察到的 cost 较低，
但缓存命中、回答长度、服务延迟和重复变化均存在，不能据三次重复推断
稳定成本优势。题目有明确字段和限制提示，身份/契约的一些负对照已经到顶；
这不等于自然优化会话中的 agent 已经会主动提出同样的问题。

## 公开回答的独立复核：固定字段还漏掉一条错误

独立审查者按调用前冻结的
[rubric](../../profile_db/research/pfdb_development/followup_review_rubric.md)，
只见匿名化的公开响应及其各自证据，不见条件、固定分数或答案索引。54 份
回答分两批逐条完成后再汇总；证据内容本身可能暴露条件差异，因此不是
完全盲法。审查同样由 agent 完成，是可复查判断，不是人工正确性认证。

发现 **11 份回答各有一条没有证据支持的已确立断言**：七条 affine/paging
混淆、三条把 512 tokens 当成观测 row，另有一条说两份 source 文件大小
不同，但 packet 明明给出二者都是 **34,840 bytes**。最后这一份的固定
字段及引用全通过，说明增加字段题目覆盖也不能替代公开解释复核。
其余 43 份没有发现此类断言，不代表部署采用已经通过。

下一实验质量分别为 **39 adequate、15 partial、0 poor**；常见不足是
缺少对照、同步判据、物理执行 row 的核算范围或输入域 authority。
提议采集数据与声称已经观测到数据分开判定，正确 unknown 被接受。

| 条件（审查完成后由主执行器解盲） | 有 unsupported claim / 18 | adequate / partial |
| --- | --- | --- |
| D0 | 2 | 11 / 7 |
| D1 | 6 | 11 / 7 |
| R | 3 | 17 / 1 |

这些是小型固定题组的描述，不能作总体成功率、显著性或策略因果结论。
尤其 D0/D1 的信息不同，R 又同时添加关系和范围说明，不能将某个分差
解释为一种通用数据表示已胜出。
逐项判据及公开解释摘录见
[复核 JSON](../../profile_db/research/pfdb_development/followup_public_review.json)
和[复核摘要](../../profile_db/research/pfdb_development/followup_public_review.md)。
字段、成本、条件映射后的审查计数、设备结果和来源 hash 的缩减账本见
[followup_summary.json](../../profile_db/research/pfdb_development/followup_summary.json)。
完整原始响应仍在忽略的 `build_output/pfdb_development/followup-v1/static-scoped/`
中，既不把隐藏推理写入报告，也不以摘要替代保存的原始诊断。

## 据此收敛 pfdb 的后续开发

本轮实现的是外部证据导出/检查原型和可重放实验，**没有给 pfdb 增加新的
顶层工具、schema 或默认展开**。现有生产 API 保持上一轮状态。理由是已经
验证的正向效果很局部，关系包同时也有未消除甚至重复出现的错误，尚不足以
支持把所有源码、生成物和派生说明默认灌给 agent。

| 优先级 | 开发对象 | 保留与省略的边界 | 后续验收 |
| --- | --- | --- | --- |
| P0 | 已有候选工件的身份与可用性分组 | 保存 source/producer/build/workload 锚点、artifact kinds、采集状态；不把 compile-only 候选伪造成完整 profiling run | 同 metadata、不同 source/IR 必须区分；错配拒绝，缺失明确 unavailable；没有设备调用 |
| P0 | 输入契约与 validation scope 分开 | 记录 authority/reference、允许的特化前提、实际 fixture 和 comparator 范围；无权威声明就 unresolved | initializer、随机测试与 PASS 都不能自动晋升输入域条款；记录失败也不擅自裁定域合法性 |
| P1 | 当前实体的短关系视图 | 按需给 unit→lane/block、chunk→allocation、space→hardware pool、输入行→观测行的关系与依据 | 在更自然的会话里纠正错误，且保留原始事实可查；不编造观测，不输出优化指令 |
| 暂缓 | 通用 IR 引擎、默认完整源码展开、批量删低频能力 | 本题的重复片段不等于底层功能冗余 | 有新增真实优化场景与质量/成本证据后，再决定默认与删除 |

P0 的最小后端接缝应是**分析已有 artifact group**，并经现有 trial/inventory
导航到当前候选的一个有界证据视图。失败候选可以有错误/编译工件而没有
profile run；不要先要求三件套 capture，也不要把候选的新 hint attach 到
旧 run 后当成同一 build。现有 attach 的文件 hash 证明内容未变，不能证明
调用者选对了 run；trial lineage 也不自动证明 source、workload 或验证覆盖。

一个可复审的导入组至少需要这些独立维度；它们是后续开发约束，不是本轮
已经发布的新 schema：

- source/parent digest、producer/toolchain、实际 compiled files digest；
  `compiled_meta` 作为其中一个 artifact，不能充当唯一 build 身份。
- workload/fixture digest、输入域 authority 与特化前提、reference/comparator
  身份、验证结果和覆盖范围；三者缺一不能相互补成“通用正确”。
- artifact kind、生成阶段、kernel/entry/source span、是否存在对应观测；
  派生关系列出输入证据和计算范围，不将静态工作量当作 measured traffic。

文件字节与 producer 声明仍不是不可伪造的编译器证明。pfdb 应验证可核对的
一致性，保留声明来源与缺失项，不能声称自动认证算法、业务域或编译责任。
runtime 已提供的时间、row、critical-path 与分析口径继续以 runtime 为准；
本轮的源码算术和资源关联需标为本地派生，不能覆盖 runtime 观测。

哪些内容现在可缩减也更明确了：若问题只问两个 build 能否匹配，指纹和
适用范围已足够，不必同时展开整段源码/IR；若只问 warned pool，已有
Right/L0B 定义已足够。相邻 chunk 的分配关系、候选的新 warning 状态和
生成 TMULS 的实际位置，则需要相应的当前工件，不能从旧摘要补猜。
这是**按当前问题裁剪 evidence**，不是删除这些查询或把全部 runtime 工具
再转发一遍。

下次放行生产默认改变前，需要新的自然优化会话：保留相同候选/计时预算，
比较 agent 能否主动定位缺失证据、选出合法候选并通过独立正确性/性能确认。
本轮没有做这一项，不能把 54 道结构化回答等同 54 次真实优化成功。

## 验证与保留

- CPU：`profile_db/tests` 与研究 fixture 共 **399 passed、11 skipped**。
  skip 是已有可选工件覆盖；没有将付费模型或 NPU 实验放入普通 CI。
- 本轮六个 Python 文件的 Ruff 检查通过；全仓 462 个受检文件的版权头通过。
- 七份相关文档的 57 个本地链接与公开 summary 一致性检查通过，
  `git diff --check` 通过。站点构建在启动前因环境缺少 `mkdocs` 模块停止；
  没有安装新依赖，因此未验证生成后的整站 HTML。
- 最终核对通过：四份原算子、三个原数据库、原 affine golden 的 16 个文件、
  本轮 21 个 source artifacts 均未改变；旧五组静态研究的 packet/runner
  快照与结果、旧优化/确认账本均匹配既存 hash。PyPTO/runtime revision
  与 tracked binary diff 保持本轮开始值。当前执行器仅有已登记的预检修正，
  不改写旧快照。
- 未提交 Git、未发布软件包、未安装任何新算子候选。

## 本轮边界

没有授权或尝试修改 runtime；没有为编译成功但未采集的候选伪造 capture。
输入域 authority、多 request oracle、实际流水重叠和更广算子上的优化收益
仍是不同问题，不能靠一份更漂亮的 evidence packet 一并解决。

后续实现：pfdb 0.6.0 已将本报告 P0 的身份、范围与缺失状态接入现有
attach/trial 导航，见[候选证据契约](pfdb-candidate-evidence.md)。这是研究结束后的
生产实现，不改写上述实验的冻结输入、评分、成本或当时未扩张接口的事实。
